import sys
import json 
import os 
import argparse 

parser = argparse.ArgumentParser(description='what are we extracting?') 
parser.add_argument('--server', dest='server', action='store_true', help='extract acr server name') 
parser.add_argument('--password', dest='password', action='store_true', help='extract acr password') 
args = parser.parse_args() 

## load acr data from stdin 
j_str_list = sys.stdin.readlines()
j = ''.join(j_str_list)
## unpack 
j = json.loads(j) 

## write to secrets/acr 
repo_dir = os.environ['repo_dir']  

if args.password: 
    token = j['passwords'][0]['value']  
    token_path = os.path.join(repo_dir, 'secret', 'acr', 'token') 
    with open(token_path, 'w') as f:
        f.write(token) 
        pass
    pass

if args.server:
    server = j['loginServer'] 
    server_path = os.path.join(repo_dir, 'secret', 'acr', 'server') 
    with open(server_path, 'w') as f: 
        f.write(server) 
        pass
    pass 
