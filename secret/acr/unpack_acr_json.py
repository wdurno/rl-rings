import sys
import json 
import os 

## load acr data from stdin 
j_str_list = sys.stdin.readlines()
j = ''.join(j_str_list)

## unpack 
j = json.loads(j) 
token = j['accessToken'] 
server = j['loginServer']

## write to secrets/acr 
repo_dir = os.environ['repo_dir']  
token_path = os.path.join(repo_dir, 'secret', 'acr', 'token') 
server_path = os.path.join(repo_dir, 'secret', 'acr', 'server') 
with open(token_path, 'w') as f:
    f.write(token) 
with open(server_path, 'w') as f: 
    f.write(server) 
