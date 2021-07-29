from connectors import mc, cc, pc
from collections import OrderedDict 
from time import time, sleep
from io import BytesIO 
import requests 
import pickle 
import types 
import torch 
import os 
import numpy as np 
import base64 

## constants 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def upload_transition(transition, manually_generated=False, retry_delay=60):
    continue_attempting = True 
    while continue_attempting: 
        try: 
            _uuid = cc.insert_game_transition(transition) 
            pc.write_transition_id(_uuid, manually_generated=manually_generated) 
            continue_attempting = False 
        except Exception as e:
            print(e)
            print('Transition upload error! Waiting '+str(retry_delay)+' before reattempting...')
            sleep(retry_delay) 
            pass
    pass

def upload_metrics():
    ## TODO decide on metrics and track them 
    pass

def sample_transitions(n=100):
    '''
    Randomly sample transitions.
    Returns list of 5 tensors if successful, otherwise `None`. 
    '''
    ## TODO what are the 5 entries? Put in doc string 
    ## pull data from db 
    uuids = pc.sample_transition_ids(n) 
    rows = cc.get_transitions(uuids) 
    ## transform into tuple of columns 
    if len(rows) == 0:
        return None 
    out = [] 
    for col in range(5): 
        ## first and third entries are dicts 
        if col in [0, 2]: 
            out.append({'pov': [], 'compass': []}) 
            #out.append({'pov': []})  
        else:
            out.append([]) 
            pass
        pass
    for row in rows:
        for col in range(5): 
            if col in [0, 2]: 
                pov = row[col]['pov']
                if type(row[col]['compass']) == dict:
                    row[col]['compass'] = row[col]['compass']['angle'] 
                compass = row[col]['compass'] 
                out[col]['pov'].append(pov) 
                out[col]['compass'].append(compass) 
            else: 
                out[col].append(row[col]) 
                pass 
            pass
    ## convert to tensors 
    for col in range(5): 
        if col in [0, 2]:
            pov = out[col]['pov'] 
            compass = out[col]['compass'] 
            out[col] = {
                    'pov': torch.from_numpy(np.stack(pov)),
                    'compass': torch.from_numpy(np.stack(compass))
                    } 
            #out[col] = out[col]['pov'] ## logic complicate asfeeds are added 
        else:
            out[col] = torch.from_numpy(np.stack(out[col])) 
            pass 
    return out 

def get_latest_model(models_dir='/models'):
    '''
    read latest model from postgres & minio
    returns:
      - `None` if no model found, `path` to local model file on disk otherwise. 
    '''
    ## load spec from postgres 
    model_id, path = pc.get_latest_model_path() 
    if model_id == 0 or path == '':
        ## no model found 
        return None 
    local_path = os.path.join(models_dir, path) 
    if os.path.isfile(local_path): 
        ## file already exists locally, no need to re-download 
        return local_path 
    ## load form MinIO
    model_blob = mc.get(path, bucket='models') 
    ## write to local disk 
    with open(local_path, 'wb') as f: 
        f.write(model_blob) 
    return local_path

def write_latest_model(model, models_dir='/models'): 
    ## save model to bytes 
    b = BytesIO() 
    torch.save(model.state_dict(), b) 
    b.seek(0) 
    model_bytes = b.read() 
    b.close() 
    ## write to minio 
    model_name = 'model-'+str(int(time()))+'.pt' 
    mc.set(model_name, model_bytes) 
    ## update latest 
    pc.set_model_path(model_name) 
    return model_name  

def download_all_transitions(): 
    uuids = cc.get_all_game_transition_uuids() 
    transition_list = cc.get_transitions(uuids) 
    return transition_list 

def __game_state_to_tensor(state: OrderedDict):
    ## only pov exists for now 
    ## more will be added with more games 
    return state['pov'] 

def __int_to_game_action(action: int): 
    'action integer to dictionary for gym usage'
    ## more logic will be added as further games are integrated 
    return {
            'attack': 1, 
            'forward': int(action == 0), 
            'back': int(action == 1), 
            'camera': np.array([30.*int(action == 2) - 30.*int(action == 3), \
                    -30.*int(action == 4) + 30.*int(action == 5)]), 
            'left': int(action == 6), 
            'right': int(action == 7), 
            'jump': int(action == 8), 
            'sneak': 0, 
            'sprint': 0, 
            'place': 'none'
            }

