## base variables 

variable "subscription_id" {
  type = string
}

variable "tenant_id" {
  type = string
}

variable "resource_group_name" { 
  type = string
  default = "rl_hypothesis_2"
} 

## phase-3 variables 

variable "number_of_ephemeral_nodes" { 
  type = number
  default = 5
}


