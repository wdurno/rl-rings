ARG BASE_IMAGE
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive \
    repo_dir=/app \
    PYTHONPATH=/app/src/python

ADD app /app

WORKDIR /app
