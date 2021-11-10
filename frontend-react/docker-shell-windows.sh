#!/bin/bash

set -e

export IMAGE_NAME="app-frontend-react"

winpty docker build -t $IMAGE_NAME -f Dockerfile.dev .
winpty docker run --rm --name $IMAGE_NAME -ti -v "$(pwd)/:/app/" -p 3000:3000 $IMAGE_NAME