## A single-node parameter server is just too slow.
## It gets network-bottlenecked. So, we have to shard
## our parameters. While I could have one tensor per 
## pod, they'll have different shapes--some quite large. 
## I'm going to optimize further by sharding parameters 
## into (approximately) equally-sized chunks. The below 
## demo illustrates the feasibility of this approach. 

import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.nn.parameter import Parameter 
from torch import autograd 

inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')
inputs = torch.from_numpy(inputs) 
targets = torch.from_numpy(targets) 

## enable row-wise access 
train_ds = TensorDataset(inputs, targets) 

## Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model
model = nn.Linear(3, 2)
print('w: ', model.weight)
print('b: ', model.bias)

mps = list(model.parameters())  
params1 = [Parameter(mps[0].detach()), Parameter(mps[1].detach())] 

params2 = [Parameter(torch.cat([Parameter(mps[0].detach().reshape((-1,))), \
        Parameter(mps[1].detach().reshape(-1,))]))] 

params3 = [Parameter(params2[0].detach()[:3]), Parameter(params2[0].detach()[3:])] 

print('before grads:')
print(mps)
print(params1) 
print(params2)
print(params3) 

## Define optimizer
## use Adam, not SGD 
opt = torch.optim.Adam(model.parameters(), lr=1e-5) 

# Define a utility function to train the model
def fit(num_epochs, model, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            # Perform gradient descent
            #loss.backward()
            grads = autograd.grad(loss, model.parameters()) 
            for g, p in zip(grads, model.parameters()): 
                p.grad = g
                pass 
            opt.step() 
            return grads 
            opt.zero_grad()
        print('loss: ', loss, 'epoch: ', epoch) 

# Train the model for 100 epochs
gg = fit(100, model, opt) 

## try to apply grads to other models 
opt2 = torch.optim.Adam(params1, lr=1e-5) 
for g, p in zip(gg, params1): 
    p.grad = g
    pass
opt2.step() 

opt3 = torch.optim.Adam(params2, lr=1e-5) 
gg_flat = torch.cat([gg[0].reshape((-1,)), gg[1].reshape((-1,))]) 
params2[0].grad = gg_flat 
opt3.step() 

opt3 = torch.optim.Adam(params3, lr=1e-5) 
for p, g in zip(params3, [gg_flat[:3], gg_flat[3:]]): 
    p.grad = g
    pass
opt3.step() 

print('grads:')
print(gg)
print(gg_flat)

print('after applying grads:') 
print(mps) 
print(params1) 
print(params2) 
print(params3) 
