## color stdout 
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e ${GREEN}setting repo dir...${NC} 
export repo_dir=$PWD

echo -e ${GREEN}configuring build...${NC} 
. ~/rl-hypothesis-2-config.sh
if [ $? != 0 ]; then
  echo -e ${RED}Please set your configuration file!${NC}
  exit 1
fi

echo -e ${GREEN}procuring base infrastructure...${NC}
cp ${repo_dir}/src/terraform/templates/base/* ${repo_dir}/src/terraform/state
cd ${repo_dir}/src/terraform/state
terraform init
. terraform-apply.sh
if [ $? != 0 ]; then
  echo -e ${RED}failed to build base infrastructure!${NC}
  exit 1
fi
cd ${repo_dir} 

echo -e ${GREEN}kubectl login...${NC}
. ${repo_dir}/src/scripts/kubectl-login.sh 

if [ $? != 0 ]; then
  echo -e ${RED}kubectl login failed!${NC}
  exit 1
fi

echo -e ${GREEN}docker login...${NC}
. ${repo_dir}/secret/acr/get_acr_access_credentials.sh

echo -e ${GREEN}postgres secret...${NC}
. ${repo_dir}/secret/postgres/make_postgres_secret.sh 
kubectl create secret generic postgres --from-file=${repo_dir}/secret/postgres/postgres-password

if [ $? != 0 ]; then
  echo -e ${RED}docker login failed!${NC}
  exit 1
fi

## switching to python3 to handle build logic 
python3 ${repo_dir}/src/python/build/build.py $@

if [ $? != 0 ]; then
  echo -e ${RED}failed to build!${NC}
  exit 1
fi

