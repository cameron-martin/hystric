#!/usr/bin/env bash

set -eo pipefail

export TFDS_DATA_DIR=/tensorflow_datasets

poetry run pip install --upgrade pip
poetry install
poetry run train