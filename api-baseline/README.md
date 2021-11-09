## Instruction

This folder holds the container and codes for running an api-service to serve two baseline models for image captioning.

Before running, copy the GCP bucket authorization file into the `secrets` folder in the parent directory, and name the file as `bucket-reader.json`.

Run `sh docker-shell.sh` to enter the container, and then run `uvicorn-server` to start the api server at `localhost:9000`.

Upon startup, it executes the function in `api/download_model.py` to scrape tokenizer and model weights from the GCP bucket into the `persistent-folder` in the parent directory. 

There are two `/predict` server created. The first one (`/predict`) serves the transformer-based model (defined in `api/model.py`), whereas the second one (`/predict2`) serves the RNN-with-attention-based model ('api/model2.py`).

The servers take an jpeg image as input and return a dictionary with generated caption in the `caption` key.
