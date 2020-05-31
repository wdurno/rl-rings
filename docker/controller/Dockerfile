FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive \
    GCLOUD_SDK_URL="https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz" \
    PATH="/opt/google-cloud-sdk/bin:/root/gsutil:${PATH}"

ADD app /app
ADD app/.boto /root/.boto

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
	openjdk-8-jdk \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/* && \
  python3 -m pip install --upgrade pip && \
  pip install cffi && \
  pip install Cython && \
  pip install lockfile && \
  python3 -m pip install -r /app/python/requirements-1.txt && \
  # gcloud 
  wget -O - -q "${GCLOUD_SDK_URL}" | tar zxf - -C /opt && \
  ln -s /lib /lib64 && \
  gcloud config set core/disable_usage_reporting true && \
  gcloud config set component_manager/disable_update_check true && \
  gcloud config set metrics/environment github_docker_image && \
  gcloud --version && \
  rm -rf /tmp/* && rm -rf /opt/google-cloud-sdk/.install/.backup && \
  # kubectl
  cd /bin && \
  curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl && \
  chmod +x kubectl && \
  # gsutil
  cd /root && \
  wget https://storage.googleapis.com/pub/gsutil.zip && \
  unzip gsutil.zip

RUN python3 -m pip install -r /app/python/requirements-2.txt 

WORKDIR /app 

CMD ["bash", "/app/entrypoint.sh"] 
