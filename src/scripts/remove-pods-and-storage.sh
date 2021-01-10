helm uninstall minio postgres simulation-storage simulation gradient-calculation $(helm list --short | grep parameter-server)
kubectl delete pvc --all --namespace=default 
