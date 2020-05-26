## configure
source config.sh
if [ -z "$RL_HYPOTHESIS_2_JOB_ID" ]; then echo "Please get a job id"; exit 1; fi
echo JOB_ID: ${RL_HYPOTHESIS_2_JOB_ID}
export RL_HYPOTHESIS_2_JOB=rlh2
export RL_HYPOTHESIS_2_INSTANCE=${RL_HYPOTHESIS_2_JOB}-${RL_HYPOTHESIS_2_JOB_ID}
export RL_HYPOTHESIS_2_DOCKER_IMAGE=${RL_HYPOTHESIS_2_DOCKER_CONTROLLER_IMAGE}
export RL_HYPOTHESIS_2_MACHINE_TYPE=e2-standard-2
## run 
source app/controller/app/scripts/spin-up-base-cluster.sh
