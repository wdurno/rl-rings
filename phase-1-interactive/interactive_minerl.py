import minerl
import gym
import readchar

env = gym.make('MineRLNavigateDense-v0') 

obs = env.reset()
done = False
net_reward = 0 

while not done:
    ## get noop action 
    action = env.action_space.noop() 
    ## get action from user 
    c = readchar.readkey() 
    if c == 'w':
        action['forward'] = 1
    elif c == 's':
        action['back'] = 1
    elif c == 'a':
        action['camera'] = [0, -30] 
    elif c == 'd':
        action['camera'] = [0, 30]
    elif c == 'j':
        action['jump'] = 1
    elif c == ' ':
        action['attack'] = 1
    ## all other actions are no-ops 
    ## apply action 
    obs, reward, done, info = env.step(action) 
    ## print rewards 
    if reward != 0:
        print('reward: '+str(reward)) 
