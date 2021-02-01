import os 
import gym
import math
import time
import uuid 
import random
import minerl
import traceback 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from multiprocessing import Process
from datetime import datetime, timedelta 
from connectors import pc, cc, mc 

from deep_net import DQN
from replay_memory import rpm
from util import get_latest_model, upload_transition, upload_metrics, sample_transitions, \
        shard_gradients, publish_grad_shards, get_all_latest_parameter_shards, \
        recombine_tensors_shards_into_parameters, shard_model_parameters, pack_shard  

## cluster role 
from cluster_config import ROLE, SIMULATION_ROLE, GRADIENT_CALCULATION_ROLE, \
        PARAMETER_SHARD_COMBINER_ROLE, SINGLE_NODE_ROLE, GRADIENT_SHARD_NUMBER, \
        TOTAL_GRADIENT_SHARDS, PARAMETER_SERVER_ROLE

## constants 
GAME = 'MineRLNavigateDense-v0' # TODO generalize to other games  
instance_id = uuid.uuid1().int >> 16 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

def time_limit(time_out):
    global start_time
    end_time = time.time()
    #print(end_time-start_time)
    if (end_time - start_time > time_out):
        return True
    else:
        return False

class Agent(object):
    
    def __init__(self, **kwargs):
        self.lr = 3e-4
        self.batch_size = 64
        self.gamma = 0.999
        self.epsilon = 0.1
        self.Vmin = -25
        self.Vmax = 25
        self.atoms = 51
        self.actions = 5
        self.policy = DQN(9, self.actions, self.atoms)
        self.target = DQN(9, self.actions, self.atoms)
        self.reward = []
        self.update_time = 0
        self.memory = rpm(125000)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr = self.lr)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)

    def get_action(self, state, test=False):
        if test:
            epsilon = 0.1
        else:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return random.randint(0, self.actions-1)
        with torch.no_grad():
            self.eval() 
            ## to device 
            pov = state['pov'] 
            compass = state['compass'] 
            pov = pov.to(dtype=torch.float, device=device) 
            compass = compass.to(dtype=torch.float, device=device) 
            ## reshape for single observation 
            pov = pov.reshape([1] + list(pov.shape)) 
            compass = compass.reshape([1] + list(compass.shape)) 
            ## predict 
            state = {
                    'pov': pov, 
                    'compass': compass
                    } 
            tmp   = (self.policy(state) * self.support).sum(2).max(1)[1] 
        return (int (tmp))

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def update_epsilon(self, rew):
        self.reward.append(rew)

        if len(self.reward) > 100:
            self.epsilon = 0.1
        elif len(self.reward) > 60:
            self.epsilon = 0.2
        elif np.sum(self.reward) > 0:
            self.epsilon = max(0.4, self.epsilon * 0.8)

    def projection_distribution(self, next_state, reward, done, gam):
        with torch.no_grad():
            batch_size = next_state['pov'].size(0)
            delta_z = float(self.Vmax - self.Vmin) / (self.atoms - 1)
            support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)

            next_dist   = self.target(next_state) * support
            next_action = next_dist.sum(2).max(1)[1]
            next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))

            #DoubleDQN
            next_dist   = self.policy(next_state).gather(1, next_action).squeeze(1)

            reward  = reward.expand_as(next_dist)
            done    = done.expand_as(next_dist)
            gam     = gam.expand_as(next_dist)
            support = support.unsqueeze(0).expand_as(next_dist)

            Tz = reward + (1 - done) * gam * support
            Tz = Tz.clamp(self.Vmin, self.Vmax)
            b  = (Tz - self.Vmin) / delta_z
            l  = b.floor().long()
            u  = b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, self.atoms)
            offset = offset.to(device)

            proj_dist = torch.zeros(next_dist.size()).to(device)

            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            #print(next_dist.sum(1))
            #print(proj_dist.sum(1))
            #input()
            return proj_dist

    def get_grads(self, game_transitions: tuple):
        ## copy parameters to device 
        self.train() 
        ## unpack transitions tuple 
        state_batch, action_batch, next_state_batch, reward_batch, done_batch, gam_batch = game_transitions
        ## q estimates 
        action_batch = action_batch.unsqueeze(1).expand(action_batch.size(0), 1, self.atoms)
        dist_pred    = self.policy(state_batch).gather(1, action_batch).squeeze(1)
        dist_true    = self.projection_distribution(next_state_batch, reward_batch, done_batch, gam_batch)
        ## calc loss and grads 
        dist_pred.data.clamp_(0.001, 0.999)
        loss = - (dist_true * dist_pred.log()).sum(1).mean()
        grads = autograd.grad(loss, self.policy.parameters())
        with torch.no_grad(): 
            q_pred = float((dist_pred * torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)).sum(1).mean()) 
        return grads, float(loss), q_pred  

    def learn(self):

        _loss = 0
        _Q_pred = 0

        if len(self.memory) < self.batch_size:
            return _loss, _Q_pred

        transitions = self.memory.sample(self.batch_size)
        grads, _loss, _Q_pred = self.get_grads(transitions) 
        
        self.apply_grads(grads)

        return _loss, _Q_pred

    def apply_grads(self, grads): 
        self.optimizer_policy.zero_grad() 
        for p, g in zip(self.policy.parameters(), grads): 
            p.grad = g 
        self.optimizer_policy.step() 
        pass 

    def train_data(self, time):
        loss = []
        Q = []
        for i in range(time):
            _loss, _Q = self.learn()
            loss.append(_loss)
            Q.append(_Q)
            self.update_time += 1
            if self.update_time % 1000 == 0:
                self.update_target()
        return np.mean(loss), np.mean(Q)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))

    def update_device(self, device=device):
        self.policy = self.policy.to(device=device)
        self.target = self.target.to(device=device)

    def train(self):
        self.policy.train()
        self.target.train()

    def eval(self):
        self.policy.eval()
        self.target.eval()

    def step(self, step, env, m_obs, m_inv, m_compass, test=False):
        TD_step = 2
        _reward = 0
        frame = 0
        done = False
        m_reward = [0 for _ in range(10)]
        m_action = [torch.tensor([0]) for _ in range(10)]
        state = [state_to(m_obs[-3:], m_compass[-3:]) for _ in range(10)]
        write_times = [] 
        while (not done) and frame < step:
            action_num = self.get_action(state[-1], test)
            obs, rew, done, info, t = envstep(env, action_num)
            _reward += rew
            frame += t

            for i in range(9):
                m_obs[i] = m_obs[i+1]
                m_inv[i] = m_inv[i+1]
                m_compass[i] = m_compass[i+1] 
                state[i] = state[i+1]
                m_reward[i] = m_reward[i+1]
                m_action[i] = m_action[i+1]


            if not done :
                m_obs[-1] = np2torch(obs['pov'])
                m_inv[-1] = obs['inventory']
                m_compass[-1] = get_compass(obs)
                state[-1] = state_to(m_obs[-3:], m_compass[-3:])
                m_reward[-1] = rew
                m_action[-1] = torch.tensor([action_num])

            if not test:
                reward, gam = 0.0, 1.0
                for i in range(TD_step):
                    reward += gam * m_reward[i-TD_step]
                    gam *= self.gamma
                reward = torch.tensor([reward])
                _done = torch.tensor([0.0])
                gam = torch.tensor([gam])
                important = reward > 0.001
                if frame >= TD_step and reward < 2.1:
                    game_transition = ([state[-TD_step-1], m_action[-TD_step], state[-1], reward, _done, gam], important) 
                    t0 = time.time() 
                    if ROLE == SINGLE_NODE_ROLE:
                        self.memory.push(*game_transition)
                    if ROLE == SIMULATION_ROLE:
                        ## upload to cassandra 
                        upload_transition(game_transition[0]) ## dropping importance  
                        pass
                    write_times.append(time.time()-t0)
                    pass
                pass

            if done and not test:
                for i in range(TD_step-1):
                    reward, gam = 0.0, 1.0
                    for k in range(TD_step-i-1):
                        reward += gam * m_reward[i-TD_step+1+k]
                        gam *= self.gamma
                    reward = torch.tensor([reward])
                    _done = torch.tensor([1.0])
                    gam = torch.tensor([gam])
                    important = frame < 17900
                    game_transition = ([state[-TD_step+i], m_action[-TD_step+i+1], state[-1], reward, _done, gam], important) 
                    t0 = time.time() 
                    if ROLE == SINGLE_NODE_ROLE: 
                        self.memory.push(*game_transition) 
                    if ROLE == SIMULATION_ROLE:
                        ## upload to cassandra 
                        upload_transition(game_transition[0]) ## dropping importance  
                        pass
                    write_times.append(time.time()-t0)
                    pass
                pass
            pass
        if not test:
            return _reward, frame, np.mean(write_times) 
        return _reward, done, np.mean(write_times) 

def action_to(num):
    act = {
        "forward": 1,
        "back": 0,
        "left": 0,
        "right": 0,
        "jump": 0,
        "sneak": 0,
        "sprint": 0,
        "attack" : 1,
        "camera": [0,0],

        "place": 0,
        #"craft": 0,
        #"equip": 1,
        #"nearbyCraft": 0,
        #"nearbySmelt": 0,
    }
    if num == 1:
        act['forward'] = 0
    elif num == 2 :
        act['jump'] = 1
    elif num == 3:
        act['camera'] = [0, -30]
    elif num == 4:
        act['camera'] = [0, 30]
    return act.copy()

def np2torch(s):
    state = torch.from_numpy(s.copy())
    return state.to(dtype=torch.float, device=device)

def state_to(pov: list, compass: list):
    ## compass 
    compass = [torch.tensor([c]) for c in compass] 
    compass = torch.cat(compass, 0)
    ## compass in [-180, 180], normalize...
    compass = compass/180. 
    ## pov 
    pov = torch.cat(pov, 2)
    pov = pov.permute(2, 0, 1)
    ## to cpu 
    pov = pov.to(torch.device('cpu')) 
    compass = compass.to(torch.device('cpu')) 
    ## format 
    state = {
            'pov': pov, 
            'compass': compass
            } 
    return state 

def envstep(env, action_num):
    reward = 0
    action = action_to(action_num)
    for i in range(4):
        obs, rew, done, info = env.step(action)
        reward += rew
        if done or action_num == 3 or action_num == 4:
            return obs, reward, done, info, i+1
    return obs, reward, done, info, 4

def get_compass(obs): 
    'Compass is buggy, throwing `None` at random.'
    compass = obs.get('compassAngle') 
    if compass is None: 
        compass = 0. ## TODO verify 0. is a good value--consider last observered 
    elif type(compass) != float:
        ## sometimes get np.array, depending on env 
        compass = float(compass) 
        pass
    return compass

def train(n_episodes):
    
    minerl_mission = GAME
    print(minerl_mission)
    env = gym.make(minerl_mission) 

    agent1 = Agent()  # treechop
    agent1.update_device()

    all_frame = 0
    rew_all = []
    i_episode = 0 
    continue_training = True 
    current_model = 'random' 
    while continue_training:
        ## get latest model 
        if ROLE == SIMULATION_ROLE:
            path = get_latest_model() 
            if path is not None: 
                ## latest model obtained
                current_model = path
                t0 = time.time() 
                agent1.load_model(path) 
                load_time = time.time() - t0
                print('loaded model: '+str(path)+', load time: '+str(load_time)+'s...') 
        ## continue loop? 
        i_episode += 1 
        if n_episodes is not None: 
            if i_episode > n_episodes: 
                continue_training = False 
        ## init env 
        env.seed((instance_id + i_episode) % 10000) 
        obs = env.reset()
        done = False
        ## simulate 
        m_obs = [np2torch(obs['pov']) for _ in range(10)] 
        m_inv = [obs['inventory'] for _ in range(10)] 
        m_compass = [get_compass(obs) for _ in range(10)] 
        _reward = 0
        frame = 0
        _reward, frame, avg_write_time = agent1.step(20000, env, m_obs, m_inv, m_compass)

        all_frame += frame
        if all_frame > 20000:
            ftime = frame // 20
        else:
            ftime = 0
        ## model fitting, if applicable 
        if ROLE == SINGLE_NODE_ROLE:
            loss, Q = agent1.train_data(ftime)
        agent1.update_epsilon(_reward)
        rew_all.append(_reward)

        ## metrics 
        if ROLE == SINGLE_NODE_ROLE: 
            print('epi %d all frame %d frame %5d Q %2.5f loss %2.5f reward %3d (%3.3f)'%\
                    (i_episode, all_frame, frame, Q, loss, _reward, np.mean(rew_all[-50:])))
        if ROLE == SIMULATION_ROLE:
            pc.write_sim_metrics(minerl_mission, current_model, _reward, frame) 
            print('epi %d all frame %d frame %5d reward %3d (%3.3f) avg_write_time: (%3.3f)'%\
                    (i_episode, all_frame, frame, _reward, np.mean(rew_all[-50:]), avg_write_time))
        pass

    # reset rpm
    if ROLE == SINGLE_NODE_ROLE: 
        agent1.memory.clear()
        agent1.save_model('train/')
    env.close()
    pass 

def grad_server(batch_size=100, model_wait_time=30, transition_wait_time=30): 
    '''
    calculate gradients for consumption by parameter server
    inputs:
      - `batch_size`: number of samples to pull per grad calculation 
      - `model_wait_time` (seconds): wait this long before re-attempting, if no model is found 
      - `transition_wait_time` (seconds): wait this long before re-attempting, if no game transitions are found
    side-effects:
      - Writes gradient UUIDs and timestamps to postgres
      - Writes gradients to cassandra 
    '''
    ### init 
    #agent = Agent() 
    while True: 
        ## pull latest model 
        t0 = time.time() 
        model_path = get_latest_model() 
        model_load_time = time.time() - t0 
        if model_path is None: 
            print('No model found. Sleeping '+str(model_wait_time)+' seconds...') 
            time.sleep(model_wait_time) 
        else:
            print('model load time: '+str(model_load_time)) 
            p = Process(target=__grad_iter, args=(model_path, batch_size, transition_wait_time,)) 
            p.start() 
            p.join()
            pass 
    pass

def __grad_iter(model_path: str, batch_size: int, transition_wait_time: int): 
    '''
    ~~use this function's scope to remedy memory leak~~
    This doesn't work. Using ACR passwords as mitigation...
    '''
    ## load model
    agent = Agent()
    agent.load_model(model_path)
    agent.update_device()
    ## sample game transitions
    t0 = time.time() 
    game_transitions = sample_transitions(batch_size) 
    sample_time = time.time() - t0 
    if game_transitions is None:
        print('Sampled transition of length zero! Sleeping '+str(transition_wait_time)+' seconds...')
        time.sleep(transition_wait_time)
    else:
        ## calculate gradeints
        grads, loss, q_pred = agent.get_grads(game_transitions)
        ## publish gradients
        grad_shards = shard_gradients(grads, TOTAL_GRADIENT_SHARDS) 
        t0 = time.time() 
        successful_writes = publish_grad_shards(grad_shards) 
        grad_shard_write_time = time.time() - t0 
        print('successfully wrote '+str(successful_writes)+' of '+str(TOTAL_GRADIENT_SHARDS)+\
                ' shards, write time: '+str(grad_shard_write_time)+', sampling time: '+str(sample_time)) 
        pc.write_grad_metrics(GAME, model_path, loss, q_pred) 
    pass 

def parameter_server(model_name: str='model', grad_wait_time: int=60, model_publish_frequency: int=60): 
    '''
    DEPRICATED!
    use `parameter_server_shard` instead
    ---
    integrate gradients, publish models
    inputs:
      - `model`: model prefix for writing to MinIO 
      - `grad_wait_time`: if no gradients are found, wait this long before reattempting 
      - `model_publish_frequency`: wait at least this long before publishing an updated model 
    side-effects:
      - writes latest model definition to Postgres
      - writes latest model blob to MinIO
    '''
    ## init
    agent = Agent() 
    agent.update_device() 
    ## get state 
    last_publish_time, last_grad_time = pc.get_parameter_server_state() 
    ## check for latest model
    model_id, model_minio_path = pc.get_latest_model_path() 
    ## cast constants 
    grad_wait_time = timedelta(seconds=grad_wait_time) 
    model_publish_frequency = timedelta(seconds=model_publish_frequency) 
    if model_id == 0 or model_minio_path == '': 
        ## no model found, publish first 
        print('No model found, writing first...') 
        model_id += 1 
        path = str(model_name) + '-' + str(model_id) + '-DQN.pkl' 
        local_path = os.path.join('/models', path) ## TODO code duplication, refactor needed, move to minio  
        agent.save_model(local_path) 
        with open(local_path, 'rb') as f:
            mc.set(path, f.read()) 
            pass
        pc.set_model_path(path) 
        pass
    while True:
        ## get new gradients 
        grad_uuid_time_df = pc.get_grad_ids_after_timestamp(last_grad_time) 
        if grad_uuid_time_df.shape[0] == 0: 
            ## wait for new grads 
            print('No new grad IDs found. Sleeping '+str(grad_wait_time)+' seconds...') 
            time.sleep(grad_wait_time.seconds) 
        else: 
            grad_uuid_list = grad_uuid_time_df.grad_id.to_list() 
            ##grads = cc.get_gradients(grad_uuid_list) # grads too big for cassandra
            grads = [] 
            for _uuid in grad_uuid_list: 
                try:
                    grads.append(mc.get_gradient(_uuid))
                except Exception as e: 
                    print('Could not get grad!') 
                    print(e) 
                    pass 
            last_grad_time = grad_uuid_time_df.timestamp.max() 
            pc.update_parameter_server_state(last_grad_time=last_grad_time) 
            if len(grads) == 0:
                ## wait for new grads 
                print('No new grads found. Sleeping '+str(grad_wait_time)+' seconds...') 
                time.sleep(grad_wait_time.seconds) 
            else: 
                ## integrate grads 
                for grad in grads:
                    agent.apply_grads(grad) 
                print('integrated '+str(len(grads))+' gradients') 
                current_time = datetime.now() 
                if current_time - last_publish_time > model_publish_frequency:
                    ## publish model 
                    model_id += 1 
                    path = str(model_name) + '-' + str(model_id) + '-DQN.pkl' 
                    local_path = os.path.join('/models', path) ## TODO global variable, put in config 
                    agent.save_model(local_path) 
                    with open(local_path, 'rb') as f:
                        mc.set(path, f.read()) 
                        pass 
                    pc.set_model_path(path) 
                    ## update time 
                    last_publish_time = current_time 
                    pc.update_parameter_server_state(last_model_publish_time=last_publish_time) 
                    print('model written: '+str(path)) 
                    pass 
    pass

def parameter_shard_combiner(publish_attempt_wait_time=90, model_name:str='model'): 
    '''
    regularly combines parameter shards and publishes to minio
    '''
    ## init
    agent = Agent() 
    agent.update_device() 
    ## get state 
    last_publish_time, last_grad_time = pc.get_parameter_server_state() 
    ## check for latest model
    model_id, model_minio_path = pc.get_latest_model_path()
    ## cast constants 
    model_publish_frequency = timedelta(seconds=publish_attempt_wait_time)  
    ## if idx == 0, publish a random model 
    if model_id == 0 or model_minio_path == '': 
        ## no model found 
        ## publish first parameter shards
        parameter_shards = shard_model_parameters(agent.policy.parameters(), TOTAL_GRADIENT_SHARDS)
        ## write each to cassandra, then publish to postgres
        for idx, parameter_shard in enumerate(parameter_shards):
            _uuid = uuid.uuid1()
            shard_b64_string = pack_shard(parameter_shard)
            cc.insert_parameter_shard_b64(_uuid, shard_b64_string)
            pc.register_parameter_server_shard(_uuid, idx)
            pass 
        ## publishing first to MinIO  
        print('No model found, writing first...') 
        model_id += 1 
        path = str(model_name) + '-' + str(model_id) + '-DQN.pkl' 
        local_path = os.path.join('/models', path) ## TODO code duplication, refactor needed, move to minio  
        agent.save_model(local_path) 
        with open(local_path, 'rb') as f:
            mc.set(path, f.read()) 
            pass
        ## increment `model_id` and set `model_minio_path` globally  
        pc.set_model_path(path) 
        print('initial model registered: '+str(path)) 
        pass
    ## forever publish new models 
    while True: 
        ## attempt to get grads 
        t0 = time.time() 
        shards = get_all_latest_parameter_shards() 
        parameter_shard_read_time = time.time() - t0 
        print('parameter shard read time: '+str(parameter_shard_read_time)) 
        parameters = agent.policy.parameters() 
        parameters = recombine_tensors_shards_into_parameters(shards, parameters) 
        now = datetime.now() 
        if parameters is not None and now - last_publish_time > model_publish_frequency: 
            ## time to publish a model 
            model_id += 1 
            path = str(model_name) + '-' + str(model_id) + '-DQN.pkl' 
            local_path = os.path.join('/models', path) ## TODO global variable, put in config 
            agent.save_model(local_path) 
            with open(local_path, 'rb') as f:
                t0 = time.time() 
                mc.set(path, f.read()) 
                model_write_time = time.time() - t0 
                print('model write time: '+str(model_write_time)) 
                pass 
            pc.set_model_path(path) 
            ## update time 
            last_publish_time = now  
            pc.update_parameter_server_state(last_model_publish_time=last_publish_time) 
            print('model written: '+str(path)) 
        else: 
            print('Waiting '+str(publish_attempt_wait_time)+' seconds before attempting to gather more shards...') 
            time.sleep(publish_attempt_wait_time) 
            pass 
    pass 

if __name__ == '__main__':
    while True: 
        try: 
            if ROLE == SINGLE_NODE_ROLE: 
                print('running single role...') 
                train(10000)
            if ROLE == SIMULATION_ROLE:
                print('running simulation role...') 
                train(None) 
            if ROLE == GRADIENT_CALCULATION_ROLE:
                print('running gradient calculation role...') 
                grad_server() 
            if ROLE == PARAMETER_SHARD_COMBINER_ROLE:
                print('running parameter shard combiner role...') 
                parameter_shard_combiner() 
            if ROLE == PARAMETER_SERVER_ROLE: 
                print('running parameter server role...') 
                parameter_server() 
        except Exception as e: 
            traceback.print_exc() 
            print(e)
            print('exception occurred, sleeping 30 seconds...') 
            time.sleep(30) 
