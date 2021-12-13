## Instruction

This folder holds the container and codes for running API servers to serve our current best model for image captioning.

Before running, copy the Before running, add the [GCP bucket authorization file](https://github.com/skgithub14/AC215_KKST/blob/main/secrets/bucket-reader.json) into the `secrets` folder in the parent directory, and name the file as `bucket-reader.json`.

Run `sh docker-shell.sh` or `docker-shell.bat` to enter the container, and then run `uvicorn_server` to start the api server at http://localhost:9000.

Upon startup, it executes the function in `api/download_model.py` to scrape tokenizer and model weights from the GCP bucket into the **persistent-folder** in the parent directory. 

The `/predict` service uses the encoder-decoder transformer model (defined in `api/model.py`) to generate caption of an image, the `/predict_prefix` service uses the prefix transformer model, the `/predict_distill` service uses the distilled prefix model, and the `/predict_rnn` service uses the RNN model.

Upon interaction with the front-end, the server takes an upladed image as input and return a dictionary with the generated caption in the form of {'caption': generated_caption}.
