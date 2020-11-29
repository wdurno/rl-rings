if [ -z ${repo_dir} ] 
then 
  echo ERROR! repo_dir not set! Run from build.sh 
  exit 1
fi

az acr login --name RlHypothesis2AzureContainerRegsitry1 --expose-token | \
python ${repo_dir}/secret/acr/unpack_acr_json.py  
