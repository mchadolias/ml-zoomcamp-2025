import os
import time
import ssl
import onnxruntime as ort
import numpy as np

from io import BytesIO
from urllib.request import Request, urlopen
from urllib.error import URLError
from PIL import Image
from typing import Dict, Any


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_CANDIDATES = [
    "hair_classifier_empty.onnx",  # from base image
    "hair_classifier_v1.onnx",  # local dev fallback
]

TARGET_SIZE = (200, 200)
TIMEOUT = 60  # increased to avoid TLS handshake timeouts
DOWNLOAD_RETRIES = 3  # retry transient network failures

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

USER_AGENT = "Mozilla/5.0 (compatible; HairClassifierLambda/1.0)"


# -----------------------------------------------------------------------------
# Model loading (cold start)
# -----------------------------------------------------------------------------

print("[INFO] Files in working directory:", os.listdir("."))

MODEL_PATH = None
for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        f"No ONNX model found. Tried {MODEL_CANDIDATES}. "
        f"Files present: {os.listdir('.')}"
    )

print(f"[INFO] Using ONNX model: {MODEL_PATH}")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------

_ssl_context = ssl.create_default_context()


def download_image(url: str) -> Image.Image:
    last_error = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=TIMEOUT, context=_ssl_context) as response:
                data = response.read()
            return Image.open(BytesIO(data))

        except URLError as e:
            last_error = e
            print(
                f"[WARN] Image download failed (attempt {attempt}/{DOWNLOAD_RETRIES}): {e}"
            )
            time.sleep(1)

    raise RuntimeError(
        f"Failed to download image after {DOWNLOAD_RETRIES} attempts: {last_error}"
    )


def prepare_image(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(TARGET_SIZE, Image.NEAREST)


def preprocess(img: Image.Image) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # HWC → CHW
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------


def predict(url: str) -> float:
    img = download_image(url)
    img = prepare_image(img)
    x = preprocess(img)

    output = session.run([output_name], {input_name: x})
    return float(np.squeeze(output[0]))


# -----------------------------------------------------------------------------
# Lambda entrypoint
# -----------------------------------------------------------------------------


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        url = event.get("url")
        if not url:
            return {"error": "Missing 'url' field"}

        if not url.startswith(("http://", "https://")):
            return {"error": "Invalid URL scheme"}

        return {"prediction": predict(url)}

    except Exception as e:
        return {"error": str(e)}


# -----------------------------------------------------------------------------
# Local execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    test_event = {
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    }
    print(lambda_handler(test_event, None))
