## color stdout 
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e ${GREEN}setting repo dir...${NC} 
export repo_dir=$PWD

echo -e ${GREEN}configuring build...${NC} 
. ~/rl-hypothesis-2-config.sh

echo -e ${GREEN}getting infrastructure...${NC}
cd ${repo_dir}/terraform
terraform init
. terraform-apply.sh
cd ${repo_dir} 

