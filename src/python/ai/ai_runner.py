
NUM_EPOCH = 50 
BATCH_SIZE = 5000
batch_size_train = 64 
batch_size_test = 128 

import os 
import minerl 
import gym 
import numpy as np
from random import shuffle 
from time import time, sleep  
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset 
import horovod.torch as hvd

from ai.util import upload_transition, sample_transitions, __int_to_game_action, write_latest_model 

## Initialize MineRL environment 
env = gym.make("MineRLTreechop-v0") 

## Initialize Horovod
hvd.init() 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #input channel 1, output channel 10
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=2)
        #input channel 10, output channel 20
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2)
        #dropout layer
        self.conv2_drop = nn.Dropout2d()
        #fully connected layer
        self.fc1 = nn.Linear(20*3*3, 50) # 20*3*3 = 180 
        self.fc2 = nn.Linear(50, 7)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 20*3*3)
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
def train(model, device, optimizer, n_iter=100, discount=.99, \
        batch_size=batch_size_train):
    'fit model on current data'
    model.train()
    for _ in range(n_iter):
        transitions = sample_transitions(batch_size) 
        optimizer.zero_grad()
        loss = __loss(model, device, transitions, discount=discount) 
        loss.backward()
        optimizer.step()
        pass
    return float(loss)  

## define sample function 
def sample(model, device, max_iter_seconds=60., halt_key=None): 
    'generate new data using latest model'
    model.eval() 
    obs = env.reset() 
    done = False 
    total_reward = 0. 
    iter_count = 0 
    continue_iterating = True 
    t_start = time()  
    while continue_iterating: 
        ## shift transition 
        obs_prev = obs 
        action_int, action_dict = __get_action(model, obs_prev, device) 
        obs, reward, done, _ = env.step(action_dict) 
        total_reward += reward 
        ## store transition  
        transition = (obs_prev, action_int, obs, reward, int(done)) 
        upload_transition(transition) 
        ## if game halted, reset 
        if done:
            obs = env.reset() 
            done = False 
            pass 
        iter_count += 1 
        if time() - t_start > max_iter_seconds:
            continue_iterating = False 
        pass 
    if iter_count == 0:
        return 0. 
    reward_rate = total_reward / iter_count 
    return reward_rate 

## define test function
def test(model, device, batch_size=batch_size_test, max_iter_seconds=120., discount=.99):
    'evaluate model against test dataset'
    model.eval() 
    losses = [] 
    continue_iterating = True 
    t_start = time() 
    while continue_iterating: 
        transitions = sample_transitions(batch_size) 
        loss = __loss(model, device, transitions, discount=discount) 
        losses.append(float(loss)) 
        if time() - t_start > max_iter_seconds: 
            continue_iterating = False 
            pass
        pass
    if len(losses) == 0:
        return 0.
    return np.mean(losses) 

def __loss(model, device, transition, discount=.99): 
    ## load tensors 
    obs_prev, action, obs, reward, done = transition 
    obs_prev = obs_prev.to(device) 
    action = action.to(device) 
    obs.to(device) 
    reward = reward.to(device) 
    done = done.to(device) 
    ## shaping data  
    obs_prev = obs_prev.permute(0, 3, 1, 2)/255.-.5 # torch has channels up-front  
    obs = obs.permute(0, 3, 1, 2)/255.-.5 
    action = action.reshape(-1, 1) # gather requires same dimensions 
    ## calculate loss 
    pred_prev, pred = model(obs_prev), model(obs) 
    pred_prev_reward = pred_prev.gather(1, action) 
    pred_reward, _ = pred.max(dim=1) 
    err = pred_prev_reward - ((1 - done)*discount*pred_reward + reward) 
    return (err*err).mean() 

def __get_action(model, single_obs, device): 
    '''translate model predictions to actions
    outputs:
     - action_int: compact representation, for storage 
     - action_dict: for use by gym 
    '''
    ## format observation 
    pov = torch.from_numpy(single_obs['pov']).to(device) 
    pov = pov.reshape(1, 64, 64, 3)  
    pov = pov.permute(0, 3, 1, 2)/255.-.5 
    ## get action 
    pred_reward = model(pov) 
    action_int = int(torch.argmax(pred_reward, 1)[0]) 
    action_dict = __int_to_game_action(action_int)  
    return action_int, action_dict 

if __name__ == '__main__': 
    rank = hvd.rank() 
    t0 = time() 
    for epoch in range(1, NUM_EPOCH + 1):
        reward_rate = sample(model, device) 
        t1 = time() - t0 
        print(f'hvd-{rank}: dt: {t1}, reward_rate: {reward_rate}') 
        train(model, device, optimizer) 
        t1 = time() - t0
        print(f'hvd-{rank}: dt: {t1}, fitting iteration complete')
        loss = test(model, device) 
        t1 = time() - t0 
        print(f'hvd-{rank}: dt: {t1}, epoch {epoch} of {NUM_EPOCH}, loss: {loss}') 
        if rank == 0: 
            write_latest_model(model) 
