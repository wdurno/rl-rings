from connectors import mc, cc, pc
from time import sleep 
import torch 
import os 

## constants 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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
    ## TODO decide on metrics and track them 
    pass

def sample_transitions(n=100):
    '''
    Randomly sample transitions, ready for `get_grads`.
    Returns list of tensors if successful, otherwise `None`. 
    '''
    ## pull data from db 
    uuids = pc.sample_transition_ids(n) 
    rows = cc.get_transitions(uuids) 
    ## transform into tuple of columns 
    if len(rows) == 0:
        return None 
    out = [] 
    for _ in range(6): 
        out.append([]) 
        pass
    for row in rows:
        for col in range(6): 
            out[col].append(row[col]) 
            pass
    ## convert to tensors 
    for col in range(6): 
        tensor = torch.stack(tuple(out[col])) 
        out[col] = tensor.to(device) 
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



