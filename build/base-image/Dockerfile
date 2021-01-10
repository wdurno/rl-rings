FROM ubuntu 

ENV DEBIAN_FRONTEND=noninteractive

ADD python-requirements /python-requirements

RUN apt-get update && \
    apt-get install -y python3-dev \
        python3-pip \
	openjdk-8-jdk \
	vim \
	curl \
	xvfb \
	x11-xserver-utils \
	python3-psycopg2 && \
    pip3 install -r /python-requirements/model-requirements.txt && \
    pip3 install -r /python-requirements/connector-requirements.txt && \
    pip3 install -r /python-requurements/parameter-server-requirements.txt 

