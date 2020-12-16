## Just learning how to pytorch 

import numpy as np
import torch

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

y = w * x + b
print(y)

# Compute gradients
y.backward() 

# Display gradients
print('dy/dw:', w.grad)
print('dy/db:', b.grad)

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')


inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print('w: ', w) 
print('b: ', b) 

def model(x):
    return x @ w.t() + b

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# Train for 100 epochs
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    print('loss: ', loss, ' at i: ', i)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5 # learning rates, lol
        b -= b.grad * 1e-5 # read a book on numerical methods or even entry calc  
        w.grad.zero_()
        b.grad.zero_()



