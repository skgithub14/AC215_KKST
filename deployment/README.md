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

## Set up Docker container for deployment
Within this folder, run `sh docker-shell.sh` in the terminal to build and enter the docker container.

Once inside the container, check the versions of these tools:
```
gcloud --version
ansible --version
kubectl version --client
```

Next, run `gcloud auth list` to check the authentication to GCP.

## Set up SSH key
First, configure OS Login for the service account:
```
gcloud compute project-info add-metadata --project <YOUR GCP_PROJECT> --metadata enable-oslogin=TRUE
```

Next, create SSH key for the service account in the **secrets** folder:
```
cd /secrets
ssh-keygen -f ssh-key-deployment
cd /app
```

Then, provide the public key to the gcloud compute instances:
```
gcloud compute os-login ssh-keys add --key-file=/secrets/ssh-key-deployment.pub
```

Copy the `username` of the output, which will be used in the next step.


## Deployment setup

