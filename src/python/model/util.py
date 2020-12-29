from connectors import mc, cc, pc
from time import sleep 
import torch 

def get_latest_model(max_retries=2):
    '''
    Get latest model as produced by the paramter server. 
    Attempt at-most `max_retries` times. 
    If failed, return `None`.
    If succeeded, return a path (`str`) to the model file. 
    '''
    ## TODO implement parameter server 
    return None 

def upload_transition(transition, retry_delay=60):
    continue_attempting = True 
    while continue_attempting: 
        try: 
            _uuid = cc.insert_game_transition(transition) 
            pc.write_transition_id(_uuid) 
            continue_attempting = False 
        except Exception as e:
            print(e)
            print('Transition upload error! Waiting '+str(retry_delay)+' before reattempting...')
            sleep(retry_delay) 
            pass
    pass

def upload_metrics():
    pass

def sample_transitions(n=100):
    'randomly sample transitions, ready for `get_grads`'
    ## pull data from db 
    uuids = pc.sample_transition_ids(n) 
    rows = cc.get_transitions(uuids) 
    ## transform into tuple of columns 
    out = [] 
    for _ in range(6): 
        out.append([]) 
        pass
    for row in rows:
        for col in range(6): 
            out[col].append(row[col]) 
            pass 
    return out 