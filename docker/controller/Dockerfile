FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

ADD app /app

# Ubuntu packages + Numpy
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
	vim \
        g++  \
        git  \
        curl  \
        cmake \
        zlib1g-dev \
        libjpeg-dev \
        xvfb \
        ffmpeg \
        xorg-dev \
        libboost-all-dev \
        libsdl2-dev \
        swig \
        python3  \
        python3-dev  \
        python3-future  \
        python3-pip  \
        python3-setuptools  \
        python3-wheel  \
        python3-scipy \
        python3-tk \
        libopenblas-base  \
        libatlas-base-dev \
        cython3  \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/* && \
  python3 -m pip install --upgrade pip && \
  pip install cffi && \
  pip install Cython && \
  pip install lockfile && \
  python3 -m pip install -r /app/python/requirements-1.txt

RUN python3 -m pip install -r /app/python/requirements-2.txt 

WORKDIR /app 

CMD ["bash", "/app/entrypoint.sh"] 
