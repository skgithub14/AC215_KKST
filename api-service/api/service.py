import os
from fastapi import FastAPI, File
from starlette.middleware.cors import CORSMiddleware
import asyncio
from tempfile import TemporaryDirectory
from api.download_model import download_test_image, download_transformer_model, download_prefix_model, download_rnn_model
# from api import transformer_model as mdl_t
# from api import prefix_model as mdl_p
# from api import RNN_model as mdl_r
from api import model as mdl


# Setup FastAPI app
app = FastAPI(
    title="API Server",
    description="API Server",
    version="v1"
)

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.on_event("startup")
async def startup():
    # Startup tasks: download test image, vectorization and model weights, CLIP model
    download_test_image()
    download_transformer_model()
    download_prefix_model()
    download_rnn_model()
    mdl.load_clip()
    mdl.load_cnn_model()

@app.get("/")
async def get_index():
    return {
        "message": "Welcome to the API Service"
    }

# Make prediction using the encoder-decoder transformer model
@app.post("/predict")
async def predict(
        file: bytes = File(...)
):
    print("predict file:", len(file), type(file))

    # Save the image
    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, "test.png")
        with open(image_path, "wb") as output:
            output.write(file)

        # Make prediction
        generated_caption = mdl.generate_caption_transformer(image_path)

    return generated_caption

# Make prediction using the prefix transformer model
@app.post("/predict_prefix")
async def predict(
        file: bytes = File(...)
):
    print("predict file:", len(file), type(file))

    # Save the image
    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, "test.png")
        with open(image_path, "wb") as output:
            output.write(file)

        # Make prediction
        generated_caption = mdl.generate_caption_prefix(image_path)

    return generated_caption

# Make prediction using the RNN model (baseline model)
@app.post("/predict_rnn")
async def predict(
        file: bytes = File(...)
):
    print("predict file:", len(file), type(file))

    # Save the image
    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, "test.png")
        with open(image_path, "wb") as output:
            output.write(file)

        # Make prediction
        generated_caption = mdl.generate_caption_rnn(image_path)

    return generated_caption
