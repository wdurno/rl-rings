helm uninstall minio postgres simulation-storage simulation gradient-calculation parameter-server 
kubectl delete pvc --all --namespace=default 
