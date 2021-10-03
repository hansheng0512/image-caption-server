import io
import base64
from PIL import Image


def base64_str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode("utf-8")
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    image = Image.open(bytesObj)
    return image