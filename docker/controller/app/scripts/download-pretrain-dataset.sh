source /app/config.sh 
gcloud auth activate-service-account $RL_HYPOTHESIS_2_SERVICE_ACCOUNT_NAME --key-file /app/service-account.json || exit 1 
cat /app/service-account.json | docker login -u _json_key --password-stdin https://${RL_HYPOTHESIS_2_DOCKER_REGISTRY_HEAD} || exit 1 
mkdir -p /dat/pre-train
gsutil cp ${RL_HYPOTHESIS_2_PRETRAIN_DATA} /dat/pre-train/pre-train.tar.gz || exit 1
cd /dat/pre-train
tar -zxvf pre-train.tar.gz || exit 1
export MINERL_DATA_ROOT=/dat/pre-train
python3 -u /app/python/load_pretrain.py 
echo Success!
while true; do echo sleeping; sleep 100; done; 
