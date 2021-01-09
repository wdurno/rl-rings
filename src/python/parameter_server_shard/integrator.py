from connectors import pc, cc, mc, postgres_connector 
from model.util import unpack_shard_b64string  
from time import sleep 
import argparse 

parser = argparse.ArgumentParser(description='integrate gradient shards into parameter shards') 
parser.add_argument('--shard-index', required=True, type=int) 

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
    pass 

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
    init_local_db() 
    shard_tensor = get_current_shard(args.shard_index) 
    ## TODO init optimizer 
    ## TODO listen for grads and integrate them indefinitely  
