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
    pip3 install --upgrade -r /python-requirements/model-requirements.txt && \
    pip3 install --upgrade -r /python-requirements/connector-requirements.txt && \
    pip3 install --upgrade -r /python-requirements/parameter-server-requirements.txt 

## mixin patch 
RUN rm /usr/local/lib/python3.8/dist-packages/minerl/env/Malmo/Minecraft/build.gradle 
ADD mixin-patch/build.gradle /usr/local/lib/python3.8/dist-packages/minerl/env/Malmo/Minecraft/build.gradle 

