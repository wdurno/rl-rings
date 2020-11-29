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

echo -e ${GREEN}Running Docker build...${NC} 

