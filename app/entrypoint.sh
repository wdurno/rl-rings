export _JOB=$1
# Override JOB if arg given 
if [ -z "$var" ]; then echo "JOB=$JOB"; else export JOB=$_JOB; fi

source config.sh 
gcloud auth activate-service-account $RL_HYPOTHESIS_2_SERVICE_ACCOUNT_NAME --key-file service-account.json 
cat service-account.json | docker login -u _json_key --password-stdin https://gcr.io/gdax-dnn 

if [ $JOB == "0-in" ]; then
	echo "Interactive mode..."
	gcloud container clusters get-credentials rlh2-$RL_HYPOTHESIS_2_JOB_ID --zone $RL_HYPOTHESIS_2_ZONE --project $RL_HYPOTHESIS_2_PROJECT
	git clone $RL_HYPOTHESIS_2_GIT_REPO
fi

if [ $JOB == "5-bu" ]; then
	echo Attempting controller image build...
	cd /app/controller
	source docker-build.sh 
fi
