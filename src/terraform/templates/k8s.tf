resource "azurerm_kubernetes_cluster" "aks-cluster" {
  name                = "rl-hypothesis-2-aks"
  location            = azurerm_resource_group.rl_hypothesis_2_resource_group.location
  resource_group_name = azurerm_resource_group.rl_hypothesis_2_resource_group.name
  dns_prefix          = "rl-hypothesis-2-aks"

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_D2_v2"
  }

  identity {
    type = "SystemAssigned"
  }
}
