## libs 
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
import argparse 

from connectors import pc 
from ai.util import upload_transition, sample_transitions, \
        __int_to_game_action, write_latest_model, get_latest_model

## params 
NUM_EPOCH = 50 
BATCH_SIZE = 5000
batch_size_train = 128  
batch_size_test = 128 
learning_rate = 0.01        
momentum = 0.5
N_OPS = 10 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
CPU = torch.device('cpu')

parser = argparse.ArgumentParser(description='configure horovod execution') 
parser.add_argument('--capture-transitions', dest='capture_transitions', default=True, help='send transitions to cassandra')  

def __parse_args():
    args = parser.parse_args() 
    ## cast bools 
    if args.capture_transitions in [False, 'False', 'FALSE', 'false']: 
        args.capture_transitions = False 
    else:
        args.capture_transitions = True 
        pass 
    return args 

## Initialize MineRL environment 
env = gym.make('MineRLNavigateDense-v0') 

## Initialize Horovod
hvd.init() 

class CNN(nn.Module):
    def __init__(self, default_device=CPU):
        super(CNN, self).__init__() 
        # params 
        self.default_device = default_device 
        # convs 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1) 
        self.conv1_bn = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1) 
        self.conv2_bn = nn.BatchNorm2d(64) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1) 
        self.conv3_bn = nn.BatchNorm2d(128) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2) 
        self.conv4_bn = nn.BatchNorm2d(256) 
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1) 
        self.conv5_bn = nn.BatchNorm2d(512) 
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1) 
        self.conv6_bn = nn.BatchNorm2d(1024) 
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1) 
        self.conv7_bn = nn.BatchNorm2d(2048) 
        self.conv8 = nn.Conv2d(2048, 4096, kernel_size=3, stride=2) 
        self.conv8_bn = nn.BatchNorm2d(4096) 
        #dropout layer
        self.conv_drop = nn.Dropout2d()
        #fully connected layer
        self.fc1 = nn.Linear(4096*1*1, 100) 
        self.fc1_bn = nn.BatchNorm1d(100) 
        self.fc2 = nn.Linear(100, 20) 
        self.fc2_bn = nn.BatchNorm1d(20) 
        self.fc3 = nn.Linear(20+1, N_OPS) # 20 for pov, +1 for vec 

    def forward(self, obs): 
        obs = self.__format_observation(obs) 
        x = obs['pov'] 
        x = self.conv1(x) 
        x = self.conv1_bn(x) 
        x = F.relu(x) 
        x = self.conv2(x) 
        x = self.conv2_bn(x) 
        x = F.relu(x) 
        x = self.conv3(x) 
        x = self.conv3_bn(x) 
        x = F.relu(x) 
        x = self.conv4(x) 
        x = self.conv4_bn(x) 
        x = F.max_pool2d(x, 2)
        x = F.relu(x) 
        x = self.conv5(x) 
        x = self.conv5_bn(x) 
        x = F.relu(x) 
        x = self.conv6(x) 
        x = self.conv6_bn(x) 
        x = F.relu(x) 
        x = self.conv7(x) 
        x = self.conv7_bn(x) 
        x = F.relu(x) 
        x = self.conv8(x) # returns shape [4096, 3, 3] 
        x = self.conv8_bn(x) 
        x = self.conv_drop(x)
        x = F.max_pool2d(x, 2) # returns shape [4096, 1, 1] 
        x = F.relu(x)
        x = x.view(-1, 4096*1*1) 
        x = self.fc1(x) 
        x = self.fc1_bn(x) 
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        x = self.fc2_bn(x) 
        x = F.relu(x) 
        v = obs['vec'] 
        x = torch.cat([x,v], dim=1) 
        x = self.fc3(x) 
        return x 
    
    def __format_observation(self, obs):
        formatted_obs = {} 
        ## format pov matrix 
        pov = obs['pov']
        if type(pov) == np.ndarray: 
            pov = torch.tensor(obs['pov'].copy()) 
        pov = pov.reshape(-1, 64, 64, 3) 
        pov = pov.permute(0, 3, 1, 2)/255.-.5
        formatted_obs['pov'] = pov 
        ## format vecs 
        n = pov.shape[0] 
        vec = torch.zeros([n, 1]) 
        ## this section will complicate as games are added 
        if 'compass' in obs: 
            if type(obs['compass']) == dict:
                ## handle the single observation case 
                obs['compass'] = obs['compass']['angle']
            compass = obs['compass']
            if type(compass) == np.ndarray: 
                compass = torch.tensor(compass.copy()) 
            vec[:,0] = compass/180. 
        formatted_obs['vec'] = vec 
        ## move to default device 
        formatted_obs['pov'] = formatted_obs['pov'].to(self.default_device) 
        formatted_obs['vec'] = formatted_obs['vec'].to(self.default_device) 
        return formatted_obs
    pass 

## create model and optimizer
model = CNN(default_device=device) 
model_path = get_latest_model() 
if model_path is not None:
    ## load latest model 
    model.load_state_dict(torch.load(model_path)) 
    pass
model.to(CPU) # mitigating horovod's gpu driver errors  
optimizer = optim.SGD(model.parameters(), 
        lr=learning_rate,
        momentum=momentum)
optimizer = hvd.DistributedOptimizer(optimizer, 
        named_parameters=model.named_parameters())  
hvd.broadcast_parameters(model.state_dict(), 
        root_rank=0) 
model.to(device) 

## define train function
def train(model, device, optimizer, n_iter=100, discount=.99, \
        batch_size=batch_size_train):
    'fit model on current data'
    n_grads_integrated = 0 
    model.train()
    for _ in range(n_iter):
        transitions = sample_transitions(batch_size) 
        loss = __loss(model, device, transitions, discount=discount, optimizer=optimizer) 
        loss.backward()
        model.to(CPU) # mitigating horovod's gpu driver errors 
        optimizer.step()
        model.to(device)
        n_grads_integrated += transitions[0]['pov'].shape[0]  
        pass
    return float(loss), n_grads_integrated 

## define sample function 
def sample(model, device, max_iter_seconds=60., capture_transitions=True, prob_random_action=0.): 
    'generate new data using latest model'
    model.eval() 
    obs = env.reset() 
    done = False 
    total_reward = 0. 
    iter_count = 0 
    captured_transitions = 0 
    continue_iterating = True 
    t_start = time()  
    while continue_iterating: 
        ## shift transition 
        obs_prev = obs 
        random_action = np.random.uniform() < prob_random_action
        action_int, action_dict = __get_action(model, obs_prev, device, random_action=random_action) 
        obs, reward, done, _ = env.step(action_dict) 
        total_reward += reward 
        ## store transition 
        transition = (obs_prev, action_int, obs, reward, int(done)) 
        if capture_transitions: 
            upload_transition(transition) 
            captured_transitions += 1 
            pass 
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
        reward_rate = 0.
    else: 
        reward_rate = total_reward / iter_count 
    return reward_rate, captured_transitions  

## define test function
def test(model, device, batch_size=batch_size_test, max_iter_seconds=30., discount=.99):
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

def __loss(model, device, transitions, discount=.99, optimizer=None): 
    '''
    calculates loss
    inputs:
     - model: instance of our CNN 
     - device: GPU or CPU 
     - transitions: 5-tuple of transition data 
     - discount: q-learning discount rate 
     - optimizer: (optimizer) If not None, calculate gradients for optimization
    outout:
     - loss: (tensor) 
    '''
    ## load tensors 
    obs_prev, action, obs, reward, done = transitions 
    obs_prev['pov'] = obs_prev['pov'].to(device) 
    obs_prev['compass'] = obs_prev['compass'].to(device) 
    action = action.to(device) 
    obs['pov'] = obs['pov'].to(device) 
    obs['compass'] = obs['compass'].to(device) 
    reward = reward.to(device) 
    done = done.to(device) 
    ## shaping data  
    action = action.reshape(-1, 1) # gather requires same dimensions 
    ## calculate loss 
    model.eval() 
    pred = model(obs)
    pred_reward, _ = pred.max(dim=1) 
    if optimizer is not None: 
        model.train() 
        optimizer.zero_grad() 
    pred_prev = model(obs_prev) 
    pred_prev_reward = pred_prev.gather(1, action) 
    err = pred_prev_reward - ((1 - done)*discount*pred_reward + reward) 
    return (err*err).mean() 

def __get_action(model, single_obs, device, random_action=False): 
    '''translate model predictions to actions
    outputs:
     - action_int: compact representation, for storage 
     - action_dict: for use by gym 
    '''
    if random_action:
        action_int = np.random.choice(N_OPS) 
    else: 
        pred_reward = model(single_obs) 
        action_int = int(torch.argmax(pred_reward, 1)[0]) 
    action_dict = __int_to_game_action(action_int)  
    return action_int, action_dict 

if __name__ == '__main__': 
    rank = hvd.rank() 
    args = __parse_args() 
    t0 = time() 
    for epoch in range(1, NUM_EPOCH + 1):
        reward_rate, captured_transitions = sample(model, device, capture_transitions=args.capture_transitions, prob_random_action=1./(epoch*epoch)) 
        t1 = time() - t0 
        print(f'hvd-{rank}: dt: {t1}, reward_rate: {reward_rate}, transitions_generated: {captured_transitions}') 
        train_loss, grads_integrated = train(model, device, optimizer) 
        t1 = time() - t0
        print(f'hvd-{rank}: dt: {t1}, train_loss: {train_loss}, grads_integrated: {grads_integrated}')
        test_loss = test(model, device) 
        t1 = time() - t0 
        print(f'hvd-{rank}: dt: {t1}, epoch {epoch} of {NUM_EPOCH}, test_loss: {test_loss}') 
        if rank == 0: 
            write_latest_model(model) 
            total_transitions = pc.get_total_transitions() 
            print(f'total_transitions: {total_transitions}') 
