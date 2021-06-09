
NUM_EPOCH = 50 
BATCH_SIZE = 5000

import minerl 
import gym 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset 
from tqdm import tqdm_notebook as tqdm
import horovod.torch as hvd

from ai.util import get_latest_model, upload_transition, sample_transitions 
from connectors import cc 

## Initialize MineRL environment 
env = gym.make("MineRLTreechop-v0") 

## Initialize Horovod
hvd.init() 

## Create dataloader, in PyTorch, we feed the trainer data with use of dataloader
## We create dataloader with dataset from torchvision, 
## and we dont have to download it seperately, all automatically done

# Define batch size, batch size is how much data you feed for training in one iteration
batch_size_train = 64 # We use a small batch size here for training
batch_size_test = 1024 #

# define how image transformed
image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
#image datasets
train_dataset = torchvision.datasets.MNIST('dataset/', 
                                           train=True, 
                                           download=True,
                                           transform=image_transform)
test_dataset = torchvision.datasets.MNIST('dataset/', 
                                          train=False, 
                                          download=True,
                                          transform=image_transform)
#data loaders
#train_loader = torch.utils.data.DataLoader(train_dataset,
#                                           batch_size=batch_size_train, 
#                                           shuffle=True)

train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank()) 

train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler) 

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test, 
                                          shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #input channel 1, output channel 10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        #input channel 10, output channel 20
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        #dropout layer
        self.conv2_drop = nn.Dropout2d()
        #fully connected layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
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
        return F.log_softmax(x, dim=1)

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
def train(model, device, train_loader, optimizer, epoch, log_interval=10000):
    'fit model on current data'
    model.train()
    tk0 = tqdm(train_loader, total=int(len(train_loader)), disable=True)
    counter = 0
    for batch_idx, (data, target) in enumerate(tk0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        counter += 1
        #tk0.set_postfix(loss=(loss.item()*data.size(0) / (counter * train_loader.batch_size)))
        pass
    pass

## define sample function 
def sample(model, device, store_observations=True, max_iter=20000): 
    'generate new data using latest model'
    obs = env.reset() 
    done = False 
    iter_counter = 0 
    while not done: 
        ## shift transition 
        obs_prev = obs 
        ## TODO use model to pick action 
        action = env.action_space.noop() 
        obs, reward, done, _ = env.step(action) 
        ## store transition  
        transition = (obs_prev, action, obs, reward, done) 
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

def RLRingsDataset(Dataset):
    'loads game transitions from Cassandra'
    def __init__(self): 
        self.__uuids = cc.get_all_game_transition_uuids() 
        pass 

    def __len__(self): 
        return len(self.__uuids) 

    def __getitem__(self, idx): 
        return cc.get_game_transition(self.__uuids[idx]) 
    pass 

def refresh_datasets():
    'Gets datasets with latest game transitions' 
    ## TODO 
    pass

if __name__ == '__main__': 
    for epoch in range(1, NUM_EPOCH + 1):
        sample(model, device)
        ## TODO model sync here to force cluster synchronization 
        ## TODO refresh dataset 
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader) 
