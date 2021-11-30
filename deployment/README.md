# Introduction

This folder contains the files to build the Docker container and Ansible scripts for deploying our Image Captioning App to web with Kubernetes on Googld Cloud Platform (GCP).

## APIs required on GCP
- Compute Engine API
- Service Usage API
- Cloud Resource Manager API
- Google Container Registry API

## GCP service accounts
For deployment, we need to set up two GCP service account.
1. `deployment` service account with required roles:
- Compute Admin
- Compute OS Login
- Container Registry Service Agent
- Kubernetes Engine Admin
- Service Account User
- Storage Admin
Once the account is created, make a json key named `deployment.json` and add it the the **secrets** folder.

2. `gcp-service` service account with one required role:
- Storage Object Viewer
Once the account is created, make a json key named `gcp-service.json` and add it the the **secrets** folder.

Note that besides these two keys, one should obtain the `bucket-reader.json` file from the team members and add it to the **secrets** folder too.

##
