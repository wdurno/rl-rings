resource "azurerm_kubernetes_cluster_node_pool" "ephemeral_node_pool" {
  name                  = "ephemeral"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks_cluster.id
  vm_size               = "Standard_F2S_v2"
  node_count            = var.number_of_ephemeral_nodes 
  
  priority              = "Spot"
  spot_max_price        = -1
  eviction_policy       = "Delete"

  tags = {
    node_type = "ephemeral"
  }
}

