helm uninstall minio postgres simulation-storage simulation gradient-calculation \
	parameter-shard-combiner $(helm list --short | grep parameter-server) 
kubectl delete pvc --all --namespace=default 
