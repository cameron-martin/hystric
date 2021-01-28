#!/usr/bin/env bash

set -eo pipefail

if [ -z "$TFDS_DATA_DIR" ]; then
    echo "You must set the TDFS_DATA_DIR environment variable"
    exit 1
fi

COMMAND_NAME=$1

if [ -z "$COMMAND_NAME" ]; then
    echo "You must specify which command to run"
    exit 1
fi

IMAGE_ID_FILE=tmp/docker-image-id
CONTAINER_ID_FILE=tmp/docker-container-id-$COMMAND_NAME

USER_ID=$(id -u $SUDO_USER)
GROUP_ID=$(id -g $SUDO_USER)

mkdir -p tmp
chown $USER_ID:$GROUP_ID tmp

if [ ! -f "$IMAGE_ID_FILE" ]; then
    docker build --iidfile "$IMAGE_ID_FILE" --build-arg user_id=$USER_ID --build-arg group_id=$GROUP_ID .
    chown $USER_ID:$GROUP_ID "$IMAGE_ID_FILE"
fi

IMAGE_ID="$(cat "$IMAGE_ID_FILE")"

if [ ! -f "$CONTAINER_ID_FILE" ]; then
    docker create -it -u $USER_ID:$GROUP_ID -v $(realpath .):/project -v $TFDS_DATA_DIR:/tensorflow_datasets -e TF_GPU_THREAD_MODE=gpu_private --workdir /project --gpus all --cidfile "$CONTAINER_ID_FILE" "$IMAGE_ID" ./bin/run-container.sh $COMMAND_NAME
    chown $USER_ID:$GROUP_ID "$CONTAINER_ID_FILE"
fi

CONTAINER_ID="$(cat "$CONTAINER_ID_FILE")"

docker start -ai "$CONTAINER_ID"