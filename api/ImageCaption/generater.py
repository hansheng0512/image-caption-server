from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import spacy
import os

from api.ImageCaption.configs import TRANSFORM


class Vocabulary:

    def __init__(self, freq_threshold=5):
        # freq_threshold check the freq word in text, if 1 , may not important to us
        # <UNK> if the word freq appear is less than threshold, it will map to <UNK>
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.spacy_eng = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.itos)

    # tokenize the caption [I love coffee] -> ["i","love","coffee"]
    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]

    # build vocab
    def build_vocabulary(self, sentence_list):
        # count each caption how many times a specific word repeated
        # if over the threshold we will include it, else ignore it
        frequencies = {}
        # start with index 4 because we have include the tagging
        idx = 4

        # self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "i"}
        # self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "i":4}

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1
                    # if the word frequence is we want,(we just need to append 1 times only)
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1  # store word to next index

    # convert text into number
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]  # if word is not in library, <UNK>
            for token in tokenized_text
        ]


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use pretrain model and fine tune last layer model
        # change to resnet and see
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # Map to embed size
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # image shape (channel,height,width) -> [3, 299, 299]
        features = self.inception(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        # Embedding layer - map word vector to better dimension space
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # captions shape: (seq_length, N) N-> batch size
        # embeddings shape: (seq_length, N,embedding_size = 256)

        embeddings = self.dropout(self.embed(captions))
        # it will take like first word to the LSTM
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):

        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    # This method use when in inference
    def caption_image(self, image, vocabulary, max_length=50):
        # image shape # batch size, channel, height, weight
        result_caption = []
        outputs = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                outputs.append(output)
                # Select the highest probabilies as the word for particular timestep 
                predicted = output.argmax(1)
                # Each predicted is 1 word
                result_caption.append(predicted.item())

                # Next word to input to lstm, repeat the loop
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]


class ImageCaptionNet(nn.Module):
    embed_size = 64
    hidden_size = 64
    freq_threshold = 5
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 250
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_saving_path = os.path.join('api/ImageCaption/data/', 'image_caption_dict.pt')
    df = pd.read_csv(os.path.join('api/ImageCaption/data/', 'captions.csv'))
    imgs = df["image"]
    captions = df["caption"]
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(captions.tolist())
    vocab_size = len(vocab)

    def __init__(self):
        super().__init__()
        self.model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers).to(self.device)
        self.model.load_state_dict(torch.load(self.model_saving_path))
        self.model.eval()

    def predict(self, image):
        test_img1 = TRANSFORM(image.convert("RGB")).unsqueeze(0)
        text = self.model.caption_image(test_img1.to(self.device), self.vocab)
        return text
