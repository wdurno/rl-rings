from build.util import run 
import os 
import json 

def refresh_keys(root, conf): 
    __get_kubeconfig(conf) 
    __get_acr_token(root, conf) 
    __get_acr_server(root, conf) 
    __upload_acr_secret_to_k8s(root) 
    pass 

def __get_kubeconfig(conf):
    tf_prefix = conf['terraform_prefix'] 
    resource_group = f'{tf_prefix}rg'
    cluster_name = f'{tf_prefix}k8s'
    cmd = f'az aks get-credentials --name {cluster_name} --resource-group {resource_group} --overwrite-existing'
    run(cmd) 
    pass 

def __get_acr_token(root, config): 
    ## get token in JSON from az cli stdout 
    tf_prefix = config['terraform_prefix'] 
    acr_name = f'{tf_prefix}acr'
    cmd = f'az acr credential show -n {acr_name} -o json'
    json_str = run(cmd, return_stdout=True) 
    ## parse JSON and save token 
    j = json.loads(json_str)
    token = j['passwords'][0]['value']  
    token_path = os.path.join(root, 'secret', 'acr', 'token') 
    with open(token_path, 'w') as f:
        f.write(token) 
        pass
    pass

def __get_acr_server(root, config): 
    ## get server in JSON from az cli stdout 
    tf_prefix = config['terraform_prefix'] 
    acr_name = f'{tf_prefix}acr' 
    cmd = f'az acr show -n {acr_name} -o json'
    json_str = run(cmd, return_stdout=True) 
    ## parse JSON and save server 
    j = json.loads(json_str) 
    server = j['loginServer'] 
    server_path = os.path.join(root, 'secret', 'acr', 'server') 
    with open(server_path, 'w') as f: 
        f.write(server) 
        pass
    pass 

def __upload_acr_secret_to_k8s(root): 
    cmd1 = f'kubectl delete secret acr-creds'
    try:
        run(cmd1) 
    except:
        ## if secret doesn't exist yet, just create a new one 
        pass
    cmd2 = 'kubectl create secret docker-registry acr-creds '+\
        f'--docker-server=$(cat {root}/secret/acr/server) '+\
        '--docker-username=horovodK8sPytorchacr '+\
        f'--docker-password=$(cat {root}/secret/acr/token)'
    run(cmd2) 
    pass

