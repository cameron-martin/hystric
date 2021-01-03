#!/usr/bin/env bash

set -eo pipefail

IMAGE_ID_FILE=tmp/docker-image-id
CONTAINER_ID_FILE=tmp/docker-container-id

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
    docker create -it -u $USER_ID:$GROUP_ID -v $(realpath .):/project --workdir /project --gpus all --cidfile "$CONTAINER_ID_FILE" "$IMAGE_ID" ./bin/run-container.sh
    chown $USER_ID:$GROUP_ID "$CONTAINER_ID_FILE"
fi

CONTAINER_ID="$(cat "$CONTAINER_ID_FILE")"

docker start -ai "$CONTAINER_ID"