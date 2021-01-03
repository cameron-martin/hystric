FROM tensorflow/tensorflow:2.4.0-gpu

ARG user_id
ARG group_id

RUN groupadd -r -g ${group_id} solver && useradd --no-log-init -m -r -u ${user_id} -g solver solver

RUN apt install -y python3-venv libsndfile-dev ffmpeg

USER solver

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/1.0.9/get-poetry.py | python

ENV PATH="/home/solver/.poetry/bin:${PATH}"