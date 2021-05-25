FROM ubuntu:18.04

ARG python=3.6

ENV TENSORFLOW_VERSION=1.14.0 \
    PYTORCH_VERSION=1.4.0 \
    TORCHVISION_VERSION=0.5.0 \
    MXNET_VERSION=1.6.0 \
    HOROVOD_VERSION=0.19.0 \
    PYTHON_VERSION=${python} \
    DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN echo "installing libraries..." && \
    apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
	openjdk-8-jdk \
	xvfb \
	x11-xserver-utils \
	python3-psycopg2 && \
    echo "installing python and fundamental packages..." && \
    if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
        apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install \
        future \
	typing \
	numpy \
	scipy \
	matplotlib \
	gym \
	minerl \
	minio \
	cassandra-driver \
	pandas && \
    pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} && \
    echo "installing openmpi..." && \
    mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi && \
    echo "installing horovod..." && \
    HOROVOD_WITH_TENSORFLOW=0 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=0 \
    pip install --no-cache-dir horovod==${HOROVOD_VERSION} && \
    echo "install ssh for mpi..." && \
    apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config && \
    ssh-keygen -b 2048 -t rsa -f ~/.ssh/id_rsa -q -N "" && \
    cp ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && \
    chmod 700 ~/.ssh/*

## these layers are most-likely to be modified during development 
ADD src /app/src

RUN pip3 install -r /app/src/python/requirements.txt

WORKDIR "/app"
