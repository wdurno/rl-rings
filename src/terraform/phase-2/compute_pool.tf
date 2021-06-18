  
resource "azurerm_kubernetes_cluster_node_pool" "compute_pool" {
  name                  = var.compute_pool_name 
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks_cluster.id
  vm_size               = var.compute_node_type
  node_count            = var.number_of_compute_nodes 
  
  priority              = "Spot"
  spot_max_price        = -1
  eviction_policy       = "Delete"

  tags = {
    node_type = "compute"
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
