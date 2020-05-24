
echo "configuring environment variables..."

echo "===="
cat config.sh 
echo "===="
source config.sh  
echo "===="
env | grep RL_HYPOTHESIS_2  
echo "===="

echo "building builder image..."

echo "===="
cp config.sh app/config.sh
cp config.sh app/controller/app/config.sh 
cp $RL_HYPOTHESIS_2_SERVICE_ACCOUNT_JSON_PATH app/service-account.json
cp $RL_HYPOTHESIS_2_SERVICE_ACCOUNT_JSON_PATH app/controller/app/service-account.json 
cat docker-build.sh 
echo "===="
bash docker-build.sh 
echo "===="
rm app/service-account.json
rm app/controller/app/service-account.json 
rm app/config.sh
rm app/controller/app/config.sh 

