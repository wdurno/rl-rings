from torch.util.data.dataset import Dataset 
from connectors import pc, cc

class ShardedDataset(Dataset): 
    'for observations stored in Cassandra and registered in postgres'

    def __init__(self, expected_len=10000, sampled_indices=None): 
        '''
        Samples `expected_len` IDs, totally according to a binomial distribution.
        Alternatively, provide `sampled_indices` directly as a list. 
        '''
        if samples_indices is not None:
            self.__sampled_indices = sampled_indices 
        else:
            self.__sampled_indices = pc.sample_transition_ids(expected_len) 
        pass 

    def __len__(self):
        return len(self.__sampled_indices) 

    def __getitem__(self, index):
        return cc.get_game_transition(self.__sampled_indices[index]) 

    def __add__(self, dataset):
        return ShardedDataset(sampled_indices=self.__sampled_indices + dataset.__sampled_indices) 
    pass 

