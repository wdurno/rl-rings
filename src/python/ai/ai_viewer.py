import sys 
import tty 
import torch 
from ai.ai_runner import model, device, sample, env
from ai.util import get_latest_model, upload_transition, __int_to_game_action  
import argparse 

parser = argparse.ArgumentParser(description='Run a viewable environment') 
parser.add_argument('--interactive-mode', dest='interactive_mode', default=False, help='manually control the session.') 

def __parse_args(): 
    args = parser.parse_args()
    ## cast bool args 
    if args.interactive_mode in [True, 'True', 'true', 'TRUE']: 
        args.interactive_mode = True 
    else:
        args.interactive_mode = False 
        pass 
    return args 

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

def interactive_mode():
    print(__help) 
    obs = env.reset() 
    total_reward = 0.
    total_iterations = 0 
    while True: 
        total_iterations += 1 
        ## get action 
        key = __wait_for_key() 
        if key in key_map.keys():
            ## action provided 
            action_int = key_map[key] 
            action_dict = __int_to_game_action(action_int) 
            ## shift transition 
            obs_prev = obs 
            obs, reward, done, _ = env.step(action_dict) 
            total_reward += reward 
            ## store transition  
            transition = (obs_prev, action_int, obs, reward, int(done)) 
            upload_transition(transition, manually_generated=True) 
            print(f'action_int: {action_int}, total_reward: {total_reward}, total_iterations: {total_iterations}') 
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

__key_help = {
        'w': 'forward',
        's': 'back', 
        'a': 'left', 
        'd': 'right', 
        'i': 'look-up',
        'k': 'look-down',
        'j': 'look-left', 
        'l': 'look-right', 
        ' ': 'jump', 
        'e': 'no-op'
        } 

__key_map = { 
        'forward': 0,
        'back': 1,
        'look-down': 2,
        'look-up': 3,
        'look-left': 4,
        'look-right': 5,
        'left': 6,
        'right': 7,
        'jump': 8, 
        'no-op': 9
        }

key_map = {} ## actual mapping 
for key in __key_help.keys():
    key_map[key] = __key_map[__key_help[key]] 
    pass 

__help = f'''Press 'h' for this help message.
Keys are mapped to `wads` and num pad.
{__key_help}
Other actions:
- n: new environment
'''

def __wait_for_key():
    return sys.stdin.read(1) 

if __name__ == '__main__': 
    args = __parse_args() 
    if args.interactive_mode:
        tty.setcbreak(sys.stdin.fileno()) ## read individual bytes from stdin without EOFs
        interactive_mode()  
    else:
        non_interactive_mode() 
