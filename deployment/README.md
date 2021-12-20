# Introduction

This folder contains the files to build the Docker container and Ansible scripts for deploying our Image Captioning App to web with Kubernetes on Googld Cloud Platform (GCP).

## APIs required on GCP
- Compute Engine API
- Service Usage API
- Cloud Resource Manager API
- Google Container Registry API

## GCP service accounts
For deployment, we need to set up two GCP service accounts.
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

Note that besides these two keys, make sure to put the `bucket-reader.json` file in the **secrets** folder too.

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


## Deployment in a GCP Compute Instance (VM)
Before deployment, go to the `inventory.yml` file, change the ansible user to the username copied from last step, and change the service account email and project name to your own.

First, build and push the Docker images for the api-service and frontend to Google Container Registry:
```
ansible-playbook deploy-docker-images.yml -i inventory.yml
```

Next, create a VM instance in GCP:
```
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present
```
Once the VM is created, get the IP address of the compute instance from GCP Console and update the appserver>hosts in the `inventory.yml` file.


Provision:
```
ansible-playbook deploy-provision-instance.yml -i inventory.yml
```

Setup Docker containers in the Compute Instance:
```
ansible-playbook deploy-setup-containers.yml -i inventory.yml
```

Setup webserver in the Compute Instance:
```
ansible-playbook deploy-setup-webserver.yml -i inventory.yml
```

Last, go to `http://<External IP>/` to see the Image Captioning App.

To delete the instnace, run
```
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=absent
```

##  Deployment in a GCP Kubernetes Cluster

Before deploying on K8s cluster, make sure the Docker images for the api-service and frontend have been pushed to Google Container Registry.

Within the deployment container, run `gcloud auth list` to check GCP authentification. 

To deploy, run
```
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present
```

To view the App, copy the `nginx_ingress_ip` from the terminal after executing the create clsuter command, and then go to `http://<YOUR INGRESS IP>.sslip.io` to see the deployed App.


To delete the cluster, run
```
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=absent
```

