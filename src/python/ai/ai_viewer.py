import sys 
import torch 
from ai.ai_runner import model, device, sample, env
from ai.util import get_latest_model, upload_transition, __int_to_game_action  
import argparse 

parser = argparse.ArgumentParser(description='Run a viewable environment') 
parser.add_argument('--interactive-mode', dest='interactive_mode', action='store_true', default=False, \
        help='manually control the session.') 

def non_interactive_mode():
    while True: 
        try:
            print('attempting to load latest model...')
            path_to_latest_model = get_latest_model() 
            model.load_state_dict(torch.load(path_to_latest_model)) 
            print('model loaded...') 
        except Exception as e:
            print('Could not load model, using local model. Reason:\n'+str(e)) 
            pass 
        print('sampling...') 
        sample(model, device) 
        pass
    pass 

def interactive_mode(max_action_val: int=6):
    print(__help) 
    obs = env.reset() 
    while True: 
        ## get action 
        key = __wait_for_key() 
        if key in '0123456789':
            ## action provided 
            action_int = int(key) 
            if action_int > max_action_val:
                action_int = max_action_val 
            action_dict = __int_to_game_action(action_int) 
            ## shift transition 
            obs_prev = obs 
            obs, reward, done, _ = env.step(action_dict) 
            total_reward += reward 
            ## store transition  
            transition = (obs_prev, action_int, obs, reward, int(done)) 
            upload_transition(transition) 
            print('transition: '+str(('[obs_prev]', action_int, '[obs]', reward, int(done)))) 
            ## if game halted, reset 
            if done:
                print('resetting environment due to `done`...') 
                obs = env.reset() 
                done = False
                pass
            pass
        elif key == 'n':
            ## reset game 
            print('resetting environment due to `n`...') 
            obs = env.reset() 
            done = False 
        else: ## including key == 'h' 
            print(__help) 
            pass 
        pass 
    pass 

__help = '''Press 'h' for this help message.
Bot actions are mapped to number keys.
Other actions:
- n: new environment
'''

def __wait_for_key():
    return sys.stdin.read(1) 

if __name__ == '__main__': 
    if args.interactive_mode:
        interactive_mode(6) # 0-6 
    else:
        non_interactive_mode() 
