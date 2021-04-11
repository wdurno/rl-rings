resource "azurerm_container_registry" "azure_container_registry_1" {
  name                     = var.acr_name
  resource_group_name      = azurerm_resource_group.rl_hypothesis_2_resource_group.name
  location                 = azurerm_resource_group.rl_hypothesis_2_resource_group.location
  sku                      = "Standard"
  admin_enabled            = true
}
