cluster_name=rl-hypothesis-2-aks
az aks get-credentials --name ${cluster_name} --resource-group ${rl_hypothesis_2_resource_group_name} --overwrite-existing
