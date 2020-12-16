## Reinforcement learning really needs to be a distributed 
## computation to be effective. Game simulation is expensive
## and just takes too much valuable time from optimization. 
## Also, gradient calculation is expensive and can be 
## further parallelized. This Python file is an experiment, 
## exploring how to separate gradient calculation from 
## gradient application using Pytorch.

import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
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

## Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5) 

# Define a utility function to train the model
def fit(num_epochs, model, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('loss: ', loss, 'epoch: ', epoch) 

# Train the model for 100 epochs
fit(100, model, opt) 

model2 = model = nn.Linear(3, 2) 
opt2 = torch.optim.SGD(model2.parameters(), lr=1e-5) 

## Distribute gradients 
def fit2(num_epochs, model, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            # Perform gradient descent, manually passing gradients 
            loss.backward() 
            grads = autograd.grad(loss, model.parameters()) 
            opt.zero_grad() 
            for p, g in zip(model.parameters(), grads): 
                p.grad.fill_(g) 
            opt.step()
        print('loss: ', loss, 'epoch: ', epoch)

fit(100, model2, opt2)  

