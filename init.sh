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
