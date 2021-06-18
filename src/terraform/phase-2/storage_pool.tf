  
resource "azurerm_kubernetes_cluster_node_pool" "storage_pool" {
  name                  = var.storage_pool_name 
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks_cluster.id
  vm_size               = var.storage_node_type
  node_count            = var.number_of_storage_nodes 
  
  priority              = "Spot"
  spot_max_price        = -1
  eviction_policy       = "Delete"

  tags = {
    node_type = "storage"
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
