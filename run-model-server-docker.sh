
# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

MODEL_NAME="mnist-classifier"
MODEL_BASE_PATH="$PWD/saved-models"


# Start TensorFlow Serving container and open the REST API port
docker run -i -t --rm -p 8501:8501 \
    -v "$MODEL_BASE_PATH/$MODEL_NAME:/models/$MODEL_NAME" \
    -e MODEL_NAME=$MODEL_NAME \
    tensorflow/serving


