FROM docker:dind

ENV GCLOUD_SDK_URL="https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz" \
    PATH="/opt/google-cloud-sdk/bin:/opt/spark/bin:/root/gsutil:${PATH}" 

RUN apk --update --no-cache add \
        bash \
        ca-certificates \
        curl \
        openssl \
        python \
        python3 \
        gettext \
        git \
        vim \
        openjdk8 && \
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

ADD app /app
ADD app/.ssh /root/.ssh
ADD app/.boto /root/.boto

WORKDIR /app

