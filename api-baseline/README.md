This folder holds the container and codes for running an api-service to serve two baseline models for image captioning.

Run `sh docker-shell.sh` to enter the container, and then run `uvicorn-server` to start the api server at `localhost:9000`.

There are two `\predict` server created. The first one (`\predict`) serves the transformer-based model, whereas the second one (`\predict2`) serves the RNN-with-attention-based model.

The servers take an jpeg image as input and return a dictionary with generated caption in the `caption` key.
