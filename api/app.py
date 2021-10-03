import flask
from flask import Flask, Response, jsonify, request

from .ImageCaption.generater import ImageCaptionNet
from .ImageCaption.utils.image import base64_str_to_PILImage
from .errors import errors

import torch
import torchvision.transforms as transforms
from torchvision import datasets

app = Flask(__name__)
app.register_blueprint(errors)

caption_generator = ImageCaptionNet()


@app.route("/")
def index():
    return Response("Hello, world!", status=200)


@app.route('/predict', methods=["POST"])
def predict():
    json_data = flask.request.json
    image = base64_str_to_PILImage(json_data["base64"])
    text = caption_generator.predict(image)
    return ' '.join(text)


@app.route("/custom", methods=["POST"])
def custom():
    payload = request.get_json()

    if payload.get("say_hello") is True:
        output = jsonify({"message": "Hello!"})
    else:
        output = jsonify({"message": "..."})

    return output


@app.route("/health")
def health():
    return Response("OK", status=200)
