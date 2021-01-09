## Sets environment constants according to configuration 

import os 

__DISTRIBUTED_ROLE = 'DISTRIBUTED_ROLE'
__GRADIENT_SHARD_NUMBER = 'GRADIENT_SHARD_NUMBER'
__TOTAL_GRADIENT_SHARDS = 'TOTAL_GRADIENT_SHARDS'

## potential role values 
SIMULATION_ROLE = 'SIMULATION' 
GRADIENT_CALCULATION_ROLE = 'GRADIENT_CALCULATION'
PARAMETER_SERVER_ROLE = 'PARAMETER_SERVER'
SINGLE_NODE_ROLE = 'PHASE_2_SINGLE_NODE'

## Determine role in cluster 
ROLE = os.environ.get(__DISTRIBUTED_ROLE, SINGLE_NODE_ROLE)

## if gradient server, get shard number and total shards 
GRADIENT_SHARD_NUMBER = os.environ.get(__GRADIENT_SHARD_NUMBER, None) 
TOTAL_GRADIENT_SHARDS = int(os.environ.get(__TOTAL_GRADIENT_SHARDS, '0')) 

