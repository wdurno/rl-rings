terraform apply \
  -auto-approve\
  -var="subscription_id=${rl_hypothesis_2_subscription_id}"\
  -var="tenant_id=${rl_hypothesis_2_tenent_id}"\
  -var="resource_group_name=${rl_hypothesis_2_resource_group_name}"\
  -var="number_of_ephemeral_nodes=${rl_hypothesis_2_ephemeral_nodes}"\
  -var="ephemeral_node_type=${rl_hypothesis_2_ephemeral_node_type}"\
  -var="number_of_storage_nodes=${rl_hypothesis_2_storage_nodes}"\
  -var="storage_node_type=${rl_hypothesis_2_storage_node_type}"
