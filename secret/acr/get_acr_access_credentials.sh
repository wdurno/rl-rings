if [ -z ${repo_dir} ] 
then 
  echo ERROR! repo_dir not set! Run from build.sh 
  exit 1
fi

az acr show -n RlHypothesis2AzureContainerRegsitry1 -o json | \
	python ${repo_dir}/secret/acr/unpack_acr_json.py --server

az acr credential show -n RlHypothesis2AzureContainerRegsitry1 -o json | \
	python ${repo_dir}/secret/acr/unpack_acr_json.py --password
