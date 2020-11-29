FROM alpine

ADD app /app

RUN apk --update --no-cache add \
    bash \
    curl \
    openssl \
    gfortran \
    gcc \
    python3-dev \
    python3-pip \
    py3-numpy \
    py3-scipy \
    git \
    vim \
    openjdk8

WORKDIR /app
