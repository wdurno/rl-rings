IMAGE_NAME=${RL_HYPOTHESIS_2_DOCKER_BUILDER_IMAGE}
docker build . -t $IMAGE_NAME --rm=false 
docker push $IMAGE_NAME
