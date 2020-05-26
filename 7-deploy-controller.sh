## configure
source config.sh 
if [ -z "$RL_HYPOTHESIS_2_JOB_ID" ]; then echo "Please get a job id"; exit 1; fi
echo JOB_ID: ${RL_HYPOTHESIS_2_JOB_ID}
export RL_HYPOTHESIS_2_JOB=k8s
export RL_HYPOTHESIS_2_INSTANCE=x${RL_HYPOTHESIS_2_JOB}-${RL_HYPOTHESIS_2_JOB_ID}
export RL_HYPOTHESIS_2_DOCKER_IMAGE=${RL_HYPOTHESIS_2_DOCKER_AI_IMAGE}
## run 
cat app/controller/app/kubernetes/controller-pod.yaml | envsubst | kubectl apply -f -
