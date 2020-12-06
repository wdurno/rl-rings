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

echo -e ${GREEN}copying code into build dir...${NC} 
cp -r ${repo_dir}/src app/src

echo -e ${GREEN}running docker build...${NC} 
DOCKER_SERVER=$(cat secret/acr/server)
DOCKER_TOKEN=$(cat secret/acr/token)
IMAGE_NAME=${DOCKER_SERVER}/phase2:v0.0.1
docker build -t ${IMAGE_NAME} .

echo -e ${GREEN}pushing docker image to resgistry...${NC}
cat ${DOCKER_TOKEN} | docker login ${DOCKER_SERVER} --username 00000000-0000-0000-0000-000000000000 --password-stdin
docker push ${IMAGE_NAME}
