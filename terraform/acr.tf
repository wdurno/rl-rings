resource "azurerm_container_registry" "azure_container_registry_1" {
  name                     = "AzureContainerRegsitry1"
  resource_group_name      = azurerm_resource_group.ha_api_resource_group.name
  location                 = azurerm_resource_group.ha_api_resource_group.location
  sku                      = "Premium"
  admin_enabled            = false
}
