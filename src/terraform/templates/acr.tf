resource "azurerm_container_registry" "azure_container_registry_1" {
  name                     = "RlHypothesis2AzureContainerRegsitry1"
  resource_group_name      = azurerm_resource_group.rl_hypothesis_2_resource_group.name
  location                 = azurerm_resource_group.rl_hypothesis_2_resource_group.location
  sku                      = "Standard"
  admin_enabled            = false
}
