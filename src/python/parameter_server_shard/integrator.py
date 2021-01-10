from connectors import pc, cc, mc, postgres_connector 
from model.util import unpack_shard_b64string, pack_shard  
from time import time, sleep 
from datetime import datetime 
import torch 
import argparse 
import uuid 

parser = argparse.ArgumentParser(description='integrate gradient shards into parameter shards') 
parser.add_argument('--shard-index', required=True, type=int) 
parser.add_argument('--learning-rate', required=False, type=float, default=1e-5) 

grad_wait_time = 30 
shard_write_time = 60 

def init_local_db(wait_interval=10): 
    '''
    Service is semi-stateful. 
    DB is a local cache only.
    '''
    local_pc = postgres_connector.PostgresConnector('0.0.0.0', os.environ['POSTGRES_SECRET'].replace('\n', '')) 
    continue_attempting = True 
    while continue_attempting: 
        try: 
            local_pc.init_grad_shard_storage() 
            continue_attempting = False 
        except Exception as e: 
            print(e) 
            print('Error initializing DB, sleeping '+str(wait_interval)+' seconds...') 
            sleep(wait_interval) 
            pass 
    return local_pc  

def get_current_shard(shard_idx, wait_interval=30): 
    while True: 
        try:
            _uuid = pc.get_latest_parameter_server_shard_uuid(shard_idx) 
            shard_b64_string = cc.get_parameter_shard_b64(_uuid) 
            shard_tensor = unpack_shard_b64string(shard_b64_string) 
            return shard_tensor
        except Exception as e:
            print(e) 
            print('Error getting initial shard, sleeping '+str(wait_interval)+' seconds...')
            sleep(wait_interval) 
            pass 
    pass 

if __name__ == '__main__': 
    args = parser.parse_args() 
    ## init local state 
    local_pc = init_local_db() 
    shard_tensor = get_current_shard(args.shard_index) 
    ## init optimizer
    param_dict = {'shard': torch.nn.parameter.Parameter(shard_tensor)} 
    opt = torch.optim.Adam(param_dict, lr=args.learning_rate) 
    ## track time for shard writing 
    last_shard_write_time = time() 
    while True:
        ## listen for grads and integrate them indefinitely 
        now = datetime.now() 
        grad_shard_b64_list = local_pc.read_grad_shard_after_timestamp(now) 
        grad_shards = [unpack_shard_b64string(b64) for b64 in grad_shard_b64_list] 
        if len(grad_shards) == 0:
            print('no grads found, sleeping '+str(grad_wait_time)+' seconds...') 
            sleep(grad_wait_time) 
        else: 
            print('integrating '+str(len(grad_shards))+' grad shard vectors...') 
            ## update param's grad 
            param_dict['shard'].grad = g 
            opt.step() 
            ## delete old grad shards from local cache 
            local_pc.delete_grad_shards_before_timestamp(now) 
            t = time() 
            if t - last_shard_write_time > shard_write_time: 
                ## write shard to cassandra and register with postgres 
                last_shard_write_time = t 
                _uuid = uuid.uuid1() 
                ## serialize 
                shard = param_dict['shard'].detach() 
                shard_b64string = pack_shard() 
                ## write to cassandra 
                cc.insert_parameter_shard_b64(_uuid, shard_b64string) 
                ## register uuid with postgres  
                pc.register_parameter_server_shard(_uuid, args.shard_index)
                print('parameter shard written, uuid: '+str(_uuid)+', shard id: '+str(args.shard_index)) 
