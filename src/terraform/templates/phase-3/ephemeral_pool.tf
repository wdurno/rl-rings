resource "azurerm_kubernetes_cluster_node_pool" "ephemeral_node_pool" {
  name                  = "ephemeral"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks_cluster.id
  vm_size               = var.ephemeral_node_type
  node_count            = var.number_of_ephemeral_nodes 
  
  priority              = "Spot"
  spot_max_price        = -1
  eviction_policy       = "Delete"

  tags = {
    node_type = "ephemeral"
  }
  
  ## required by azure, added regardless of configuration  
  ## adding here to avoid redeployments 
  node_labels = {
    "kubernetes.azure.com/scalesetpriority" = "spot"
  } 
  node_taints = [
    "kubernetes.azure.com/scalesetpriority=spot:NoSchedule",
  ]
}

