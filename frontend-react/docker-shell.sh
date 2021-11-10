#!/bin/bash

set -e

<<<<<<< HEAD
export IMAGE_NAME="app-frontend-react"
=======
export IMAGE_NAME="image-captioning-app-frontend-react"
>>>>>>> 30cf693246187e9ad68f38085b6ab9b807e21746

docker build -t $IMAGE_NAME -f Dockerfile.dev .
docker run --rm --name $IMAGE_NAME -ti -v "$(pwd)/:/app/" -p 3000:3000 $IMAGE_NAME