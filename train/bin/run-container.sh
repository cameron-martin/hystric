#!/usr/bin/env bash

set -eo pipefail

COMMAND_NAME=$1

export TFDS_DATA_DIR=/tensorflow_datasets

poetry run pip install --upgrade pip
poetry install
poetry run $COMMAND_NAME