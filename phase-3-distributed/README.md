# phase 3: distributed mode 

Reinforcement learning software is broken into services, for more-efficient, distributed computing. 

## service overview 

- **parameter server**: 
  - Accept and queue gradients as HTTP POST data. 
  - Integrate gradients into latest model. 
  - Publish latest model to blob storage. 
  - Provide latest model URL upon HTTP GET request. 
- **blob storage**: 
  - Provide robust, highly-available blob storage. 
- **simulation**: 
  - Forever iterate, simulating game play, producing data for model fitting. 
  - Per iteration, load latest model. 
  - Per iteration, simulate a round of game play. 
  - Per iteration, save simulation data and statistics to simulation storage. 
  - Per iteration, save summarizing statistics to metric storage. 
- **simulation storage**: 
  - Provide a highly-available, robust, key-value storage. 
  - Store game transitions. 
- **metric storage**: 
  - Provide structured storage for game statistics. 
- **gradient calculation**:
  - Forever iterate, pulling random transition samples from simulation storage, producing gradients. 
  - Push gradients to the parameter server for model integration. 

## design choices 

- This architecture is cost-effective. As a cost-cutting measure, stateful sets are used less-frequently than normal. This is fine because a large volume of generated data is only temporarily useful and robust storage architectures are used. State-saving tools are provided, allowing for snap-shots. 

