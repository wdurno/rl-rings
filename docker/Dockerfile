FROM nvidia/cuda:11.1.1-base-ubuntu20.04

ENV TENSORFLOW_VERSION=1.14.0 \
    PYTORCH_VERSION=1.4.0 \
    TORCHVISION_VERSION=0.5.0 \
    MXNET_VERSION=1.6.0 \
    HOROVOD_VERSION=0.19.0 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/src/python:/usr/lib/python3/dist-packages \
    CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/root/miniconda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN echo "installing cudnn via miniconda..." && \
    apt-get update && apt-get install -y \
        curl \
        ca-certificates \
        sudo \
        git \
        bzip2 \
        libx11-6 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh && \
    conda install -y python==3.8.3 && \
    conda clean -ya && \
    echo "CUDA 11.1-specific steps..." && \
    conda install -y -c conda-forge cudatoolkit=11.1.1 && \
    conda install -y -c pytorch \
        "pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0" \
        "torchvision=0.9.1=py38_cu111" && \
    conda clean -ya

RUN echo "installing libraries..." && \
    apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++ \
        vim \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openjdk-8-jdk \
        azure-cli \
        tmux \
        xvfb \
        x11-xserver-utils \
        fluxbox \
        x11vnc \
        python3-psycopg2 \
        python3-distutils && \
    echo "installing python and fundamental packages..." && \
    pip install --upgrade pip && \
    pip install setuptools && \
    pip install \
        scikit-build \
        future \
        typing \
        numpy \
        scipy \
        matplotlib \
        gym \
        minerl \
        minio \
        Jinja2 \
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
    echo "writing X11 cookie for secure screen sharing..." && \
    openssl rand -hex 32 > /root/.Xauthority 

## pip needs a refresh 
RUN echo "installing horovod..." && \
    HOROVOD_WITH_PYTORCH=1 \
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

RUN pip install -r /app/src/python/requirements.txt

WORKDIR "/app"
