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
  default = 2
}

variable "compute_node_type" { 
  type = string
  default = "Standard_F2S_v2"
} 

