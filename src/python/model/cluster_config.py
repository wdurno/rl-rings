## Sets environment constants according to configuration 

import os 

__DISTRIBUTED_ROLE = 'DISTRIBUTED_ROLE'

## potential role values 
SIMULATION_ROLE = 'SIMULATION' 
GRADIENT_CALCULATION_ROLE = 'GRADIENT_CALCULATION'
PARAMETER_SERVER_ROLE = 'PARAMETER_SERVER'
SINGLE_NODE_ROLE = 'PHASE_2_SINGLE_NODE'

## Determine role in cluster 
ROLE = os.environ.get(__DISTRIBUTED_ROLE, SINGLE_NODE_ROLE)

