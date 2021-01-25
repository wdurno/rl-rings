import torch 
from torch import Tensor
from torch import nn 
import torch.nn.functional as F 
from typing import Tuple 

class NNet(nn.Module): 
    def __init__(self):
        super(NNet, self).__init__() 
        self.layer1 = nn.Linear(2+1,5)
        self.layer2 = nn.Linear(5,1) 
        pass 

    def forward(self, x: Tuple): 
        x = torch.cat(x, dim=1) 
        x = self.layer1(x) 
        x = F.relu(x) 
        x = self.layer2(x) 
        return F.softmax(x) 
    pass 

def build_data(dat):
    '''
    Data must be list of tensors of floats.
    Example: [(1., [2., 3.]), (4., [5., 6.])] 
    '''
    return tuple(torch.tensor(t) for t in data)

if __name__ == '__main__': 
    data = (
            [
                [2.],
                [3]],
            [
                [3., 4],
                [5, 6]] 
            )
    x = build_data(data) 
    nnet = NNet() 
    y = nnet(x) 
    print(y) 
