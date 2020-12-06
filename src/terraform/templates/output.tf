
output "acr_login_server" {
  value       = azurerm_container_registry.azure_container_registry_1.login_server
  description = "Azure container registry login server"
}

