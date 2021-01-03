#!/usr/bin/env bash

set -eo pipefail

poetry run pip install --upgrade pip
poetry install
poetry run train