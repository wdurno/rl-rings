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

BUILD_DIR=${repo_dir}/build/base-image

echo -e ${GREEN}copying code into build dir...${NC} 
cp ${repo_dir}/src/python/model/requirements.txt ${BUILD_DIR}/python-requirements/model-requirements.txt 
cp ${repo_dir}/src/python/connectors/requirements.txt ${BUILD_DIR}/python-requirements/connector-requirements.txt 

echo -e ${GREEN}running docker build...${NC} 
DOCKER_SERVER=$(cat ${repo_dir}/secret/acr/server)
DOCKER_TOKEN=$(cat ${repo_dir}/secret/acr/token)
IMAGE_NAME=${DOCKER_SERVER}/base-image:v0.0.1
docker build -t ${IMAGE_NAME} ${BUILD_DIR}

if [ $? != 0 ]; then
    echo -e ${RED}failed to build!${NC}
    exit 1
fi

echo -e ${GREEN}pushing docker image to resgistry...${NC}
echo ${DOCKER_TOKEN} | docker login ${DOCKER_SERVER} --username 00000000-0000-0000-0000-000000000000 --password-stdin
docker push ${IMAGE_NAME}
