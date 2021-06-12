
NUM_EPOCH = 50 
BATCH_SIZE = 5000
batch_size_train = 64 
batch_size_test = 1024 

import minerl 
import gym 
from random import shuffle 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset 
import horovod.torch as hvd

from ai.util import get_latest_model, upload_transition, sample_transitions 

## Initialize MineRL environment 
env = gym.make("MineRLTreechop-v0") 

## Initialize Horovod
hvd.init() 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #input channel 1, output channel 10
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1)
        #input channel 10, output channel 20
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        #dropout layer
        self.conv2_drop = nn.Dropout2d()
        #fully connected layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 7)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        return x 

## create model and optimizer
learning_rate = 0.01
momentum = 0.5
device = "cpu"
model = CNN().to(device) #using cpu here
optimizer = optim.SGD(model.parameters(), 
        lr=learning_rate,
        momentum=momentum)
optimizer = hvd.DistributedOptimizer(optimizer, 
        named_parameters=model.named_parameters()) 
hvd.broadcast_parameters(model.state_dict(), 
        root_rank=0) 

## define train function
def train(model, device, optimizer, n_iter=100, discount=.99):
    'fit model on current data'
    model.train()
    for _ in range(n_iter):
        transition = sample_transitions(batch_size_train) 
        optimizer.zero_grad()
        loss = __loss(model, device, transition, discount=discount) 
        loss.backward()
        optimizer.step()
        pass
    pass

## define sample function 
def sample(model, device, max_iter=20000): 
    'generate new data using latest model'
    model.eval() 
    obs = env.reset() 
    done = False 
    iter_counter = 0 
    while not done: 
        ## shift transition 
        obs_prev = obs 
        ## TODO use model to pick action 
        action = env.action_space.noop() 
        obs, reward, done, _ = env.step(action) 
        action = 0 
        ## store transition  
        transition = (obs_prev, action, obs, reward, int(done)) 
        upload_transition(transition) 
        ## if game halted, reset 
        if done:
            obs = env.reset() 
        ## do not loop forever 
        iter_counter += 1 
        if iter_counter > max_iter:
            done = True 
        pass 
    pass 

## define test function
def test(model, device, test_loader):
    'evaluate model against test dataset'
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    pass

def __loss(model, device, transition, discount=.99): 
    ## load tensors 
    obs_prev, action, obs, reward, done = transition 
    obs_prev = obs_prev.to(device) 
    action = action.to(device) 
    obs.to(device) 
    reward = reward.to(device) 
    done = done.to(device) 
    ## torch channels different from tensorflow, permute required 
    obs_prev = obs_prev.permute(0, 3, 1, 2)/255. 
    obs = obs.permute(0, 3, 1, 2)/255. 
    ## calculate loss 
    pred_prev, pred = model(obs_prev), model(obs) 
    pred_prev_reward = pred_prev.gather(1, action) 
    pred_reward, _ = pred.max(dim=1) 
    err = pred_prev_reward - ((1 - done)*discount*pred_reward + reward) 
    return (err*err).mean() 

if __name__ == '__main__': 
    for epoch in range(1, NUM_EPOCH + 1):
        sample(model, device)
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader) 
