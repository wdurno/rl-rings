## color stdout 
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e ${GREEN}verifying repo_dir...${NC} 
if [ -z ${repo_dir} ]
then
    echo ERROR! repo_dir not set! Run from build.sh
    exit 1
fi

echo -e ${GREEN}configuring build...${NC} 
. ~/rl-hypothesis-2-config.sh 
BUILD_DIR=${repo_dir}/build/ai

echo -e ${GREEN}copying code into build dir...${NC} 
cp -r ${repo_dir}/src ${BUILD_DIR}/app/src

echo -e ${GREEN}running docker build...${NC} 
DOCKER_SERVER=$(cat ${repo_dir}/secret/acr/server)
DOCKER_TOKEN=$(cat ${repo_dir}/secret/acr/token)
IMAGE_NAME=${DOCKER_SERVER}/ai:${rl_hypothesis_2_ai_image_tag}
echo ${DOCKER_TOKEN} | docker login ${DOCKER_SERVER} --username RlHypothesis2AzureContainerRegsitry1 --password-stdin
docker build -t ${IMAGE_NAME} ${BUILD_DIR} --build-arg BASE_IMAGE=${DOCKER_SERVER}/base-image:v0.0.1

if [ $? != 0 ]; then
    echo -e ${RED}failed to build!${NC}
    exit 1
fi

echo -e ${GREEN}pushing docker image to resgistry...${NC}
docker push ${IMAGE_NAME}
