_JOB=$1
# Override JOB if arg given
if [ -z "$_JOB" ]; then echo "JOB=$JOB"; else export JOB=$_JOB; fi

source /app/config.sh
gcloud auth activate-service-account $RL_HYPOTHESIS_2_SERVICE_ACCOUNT_NAME --key-file service-account.json || exit 1

if [ $JOB == "8-ac" ]; then
	echo "Run actor critic..."
	cd /app/python
	python3 -u actor_critic.py 
fi

echo "Interactive mode..."
while true; do echo sleeping; sleep 100; done; 

