## Instruction

This folder holds the container and codes for running API servers to serve two baseline models for image captioning.

Before running, copy the GCP bucket authorization file into the `secrets` folder in the parent directory, and name the file as `bucket-reader.json`.

Run `sh docker-shell.sh` to enter the container, and then run `uvicorn_server` to start the api server at `localhost:9000`.

Upon startup, it executes the function in `api/download_model.py` to scrape tokenizer and model weights from the GCP bucket into the `persistent-folder` in the parent directory. 

The `/predict` service uses the transformer-based model (defined in `api/transformer_model.py`) to generate caption of an image.

Upon interaction with the front-edn, the server takes an upladed image as input and return a dictionary with the generated caption in the form of {'caption': generated_caption}.
