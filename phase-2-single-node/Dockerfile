FROM ubuntu 

ENV DEBIAN_FRONTEND=noninteractive \
    repo_dir=/app

ADD app /app

RUN apt-get update && \
    apt-get install -y python3-dev \
        python3-pip \
	openjdk-8-jdk \
	vim \
	xvfb \
	x11-xserver-utils && \
    pip3 install -r /app/src/python/model/requirements.txt 

WORKDIR /app
