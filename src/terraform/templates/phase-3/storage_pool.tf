resource "azurerm_kubernetes_cluster_node_pool" "storage_node_pool" {
  name                  = "storage"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks_cluster.id
  vm_size               = var.storage_node_type
  node_count            = var.number_of_storage_nodes 
  
  tags = {
    node_type = "storage"
  }
}

