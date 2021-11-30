AC215-KKST
==============================

This is the repository for the final project of Harvard AC215 - Advanced Pratcial Data Science (DevOps for Deep Learning), Fall 2021. 

### Team Members
- Matthew Stewart 
- Steve Knapp 
- Al-Muataz Khalil 
- Shih-Yi Tseng 
- Ed Bayes

# Topic: Image Captioning

In this project, we trained a deep learning model for image captioning, and built a web-based AI App that allows users to upload images in the frontend and generate captions with the model. 

## Model description

Our image captioning model is a transformer-based model. It consists of an encoder part based on extracted features of an input image, and a decoder part that generates caption. 

Schematic of our model:
![Schematic of the model](src/model_schematics.png)


For image feature extraction, we use the ViT-B/16 image encoder of the OpenAI CLIP model as the feature extractor, which outputs a 512-dim embedding for an input image. The CLIP model was pre-trained to minimize contrastive loss between a large dataset with matching images and captions, which learns a latent embedding that represents details of an image and its corresponding langauge descritpion.
- Read about CLIP: https://openai.com/blog/clip/
- Github for CLIP: https://github.com/openai/CLIP

For the transformer, we implemented 2 extra encoder blocks (since the CLIP embedding is already an output of a visual transformer, ViT-B/16), together with 6 decoder blocks to generate captions. The embedded dimension for both image feature and tokenized text is 512, and each attention block in the encoder/decoder has 10 attention heads. The model was trained on the [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k) and [MS-COCO](https://cocodataset.org/#home) datasets with ~600k image-caption pairs. 

Some example captions generated on images in our test data from Flickr8k and MS-COCO datasets.
![Examples of captions](src/example_captions.png)

For details of the model, please refer to this [Colab notebook](https://github.com/skgithub14/AC215_KKST/blob/main/notebooks/Transformer_based_image_captioning_with_CLIP_embedding.ipynb) in the notebook folder.




## Key components

The three key components of the App are

- **api-service**: contains codes for the models and API server
- **frontend-react**: contains codes for the React frontend
- **deployment**: contains scripts for deploying the App on Google Cloud Platform (GCP)

## Setup
### API

The `api-service` folder holds the files to set up the Docker container and pipenv virtual environment, as well as Python codes (inside `api` subfolder) for running the API server to serve our image captioning model.

Before running, add the GCP bucket authorization file into the `secrets` folder in the parent directory, and name the file as `bucket-reader.json`.

Change directory back to this folder, run `sh docker-shell.sh` to build and start the container, and then run `uvicorn_server` to start the API server at `localhost:9000`.

Upon startup, it executes the function in `api/download_model.py` to scrape tokenizer and model weights from the GCP bucket into the `persistent-folder` in the parent directory. 

The `/predict` service uses the transformer-based model (defined in `api/transformer_model.py`) to generate caption of an image.

Upon interaction with the frontend, the server takes an upladed image as input and return a dictionary with the generated caption in the form of {'caption': generated_caption}.


### Frontend


### Deployment


# Project Organization
------------
      .
      ├── LICENSE
      ├── Makefile
      ├── README.md
      ├── api-service
      ├── frontend-react
      ├── deployment
      ├── api-baseline
      ├── frontend-simple
      ├── data
      ├── models
      ├── notebooks
      ├── references
      ├── requirements.txt
      ├── setup.py
      ├── src
      │   ├── __init__.py
      │   └── build_features.py
      ├── submissions
      │   ├── milestone1_KKST
      │   ├── milestone2_KKST
      │   ├── milestone3_KKST
      │   └── milestone4_KKST
      └── test_project.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
