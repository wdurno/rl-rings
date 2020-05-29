_JOB=$1
# Override JOB if arg given
if [ -z "$_JOB" ]; then echo "JOB=$JOB"; else export JOB=$_JOB; fi

if [ $JOB == "0-in" ]; then
        echo "Interactive mode..."
fi

if [ $JOB == "8-ac" ]; then
        echo "Run actor critic..."
	cd /app/python
	python3 -u actor_critic.py 
fi

