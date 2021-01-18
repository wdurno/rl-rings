from connectors import mc, cc, pc
from time import sleep
import requests 
import types 
import torch 
import os 
import numpy as np 
import base64 
import grequests 

## constants 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

from model.cluster_config import TOTAL_GRADIENT_SHARDS 

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

def __split_tensor(tensor, n):
    'split a 1-dim tensor into n parts'
    k, m = divmod(tensor.shape[0], n) 
    return (tensor[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)) 

def shard_tensor_list(tensor_list, n_shards: int): 
    'flatten tensors and break into `n_shards`'
    ## concat 
    flat_tensors = [] 
    for t in tensor_list: 
        t = t.reshape((-1,)) 
        flat_tensors.append(t) 
        pass 
    flat_tensor = torch.cat(flat_tensors) 
    ## shard 
    shards = __split_tensor(flat_tensor, n_shards) 
    return shards

def shard_model_parameters(parameters, n_shards: int): 
    'flatten parameter tensors and break into `n_shards`' 
    tensor_list = [p.detach() for p in parameters] 
    return shard_tensor_list(tensor_list, n_shards) 

def shard_gradients(grads, n_shards: int): 
    'flatten grads and break into `n_shards`'
    return shard_tensor_list(grads, n_shards) 

def publish_grad_shards(shards: list): 
    'write to parameter shard servers'
    if isinstance(shards, types.GeneratorType): 
        shards = [shard for shard in shards]  
    n_shards = len(shards) 
    async_requests = [] 
    for i in range(n_shards): 
        b64string = pack_shard(shards[i]) 
        r = grequests.post('http://parameter-server-shard-'+str(i)+':8080/', data=b64string.encode()) 
        async_requests.append(r) 
        pass 
    responses = grequests.map(async_requests) 
    failed_responses = []
    for r in responses: 
        if r is None: 
            failed_responses.append(r) 
            print('No response recieved!') 
        elif r.status_code != 200:
            failed_responses.append(r) 
            print(r.text)
    for failed_response in failed_responses: 
        print('shard publish failed!') 
        print(failed_response)
        if failed_response is not None: 
            print(failed_response.text) 
        pass
    ## return number of successfully published gradients 
    return len(responses) - len(failed_responses) 

def pack_shard(shard_tensor): 
    'tensor -> b64string'
    np_array_bytes = shard_tensor.detach().numpy().tobytes() 
    b64string = base64.b64encode(np_array_bytes).decode()  
    return b64string 

def unpack_shard_b64string(shard_b64string): 
    'b64string -> tensor'
    np_array_bytes = base64.b64decode(shard_b64string.encode()) 
    shard_tensor = torch.from_numpy(np.frombuffer(np_array_bytes, dtype=np.float32))
    return shard_tensor 

def get_all_latest_parameter_shards(): 
    shard_uuids = pc.get_all_latest_parameter_server_shard_uuids() 
    shard_b64strings = cc.get_parameter_shard_b64strs(shard_uuids) 
    shards = [unpack_shard_b64string(b64str) for b64str in shard_b64strings] 
    return shards 

def recombine_tensors_shards_into_parameters(tensor_shards, parameters: list): 
    '''
    Read through `flat_tensor` assigning values to each parameter.
    Assignment is in-place! 
    '''
    if isinstance(parameters, types.GeneratorType): \
        parameters = [p for p in parameters] 
    if len(tensor_shards) != TOTAL_GRADIENT_SHARDS: 
        print(f'Error: `len(tensor_shards) != {TOTAL_GRADIENT_SHARDS}`!') 
        return None 
    ## tensor shards arrive in approximately equally-sized arrays 
    flat_tensor = torch.cat(tensor_shards) 
    ## extract 
    cursor = 0 
    for p in parameters: 
        ## get total floats in tensor 
        parameter_size = p.detach().reshape((-1,)).shape[0] 
        ## get tensor shape 
        parameter_shape = p.shape 
        ## extract parameter-sized tensor and allocate it into parameter 
        flat_parameter_tensor = flat_tensor[cursor:(cursor + parameter_size)] 
        parameter_tensor = flat_parameter_tensor.reshape(parameter_shape) 
        p.data = parameter_tensor 
        ## increment cursor 
        cursor += parameter_size 
        pass 
    return parameters 

