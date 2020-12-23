FROM ubuntu 

ENV DEBIAN_FRONTEND=noninteractive \
    repo_dir=/app \
    PYTHONPATH=/app/src/python

ADD app /app

RUN apt-get update && \
    apt-get install -y python3-dev \
        python3-pip \
	openjdk-8-jdk \
	vim \
	curl \
	xvfb \
	x11-xserver-utils \
	python3-psycopg2 && \
    pip3 install -r /app/src/python/model/requirements.txt && \
    pip3 install -r /app/src/python/connectors/requirements.txt 

WORKDIR /app
