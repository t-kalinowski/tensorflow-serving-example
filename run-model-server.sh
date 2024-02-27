#!/usr/bin/bash

## On Ubuntu, run
# echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
# curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
# apt-get update && apt-get install tensorflow-model-server

## make sure that "saved-models" exists in the current working directory
## if not, run ./make-model.R

MODEL_NAME="mnist-classifier"
MODEL_BASE_PATH="$PWD/saved-models"
tensorflow_model_server \
  --model_name=$MODEL_NAME \
  --model_base_path="$MODEL_BASE_PATH/$MODEL_NAME" \
  --port=0 \
  --rest_api_port=8501
