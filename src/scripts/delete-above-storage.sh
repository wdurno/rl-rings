helm uninstall simulation gradient-calculation \
	parameter-shard-combiner $(helm list --short | grep parameter-server) 
