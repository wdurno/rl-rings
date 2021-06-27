## phase-1 variables 

variable "subscription_id" {
  type = string
}

variable "tenant_id" {
  type = string
}

variable "resource_group_name" { 
  type = string
  default = "rg"
} 

variable "acr_name" { 
  type = string 
  default = "acr" 
} 

variable "k8s_name" { 
  type = string
  default = "k8s"
} 

## phase-2 variables 

variable "compute_pool_name" { 
  type = string 
  default = "compute" 
} 

variable "number_of_compute_nodes" { 
  type = number
  default = 4
}

variable "compute_node_type" { 
  type = string
  default = "standard_e2a_v4"
} 

variable "storage_pool_name" {
  type = string
  default = "storage"
}

variable "number_of_storage_nodes" {
  type = number
  default = 4
}

variable "storage_node_type" {
  type = string
  default = "Standard_A2M_v2"
}

