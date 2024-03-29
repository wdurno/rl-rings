import argparse
import os
from time import sleep
from build.util import run 
import requests

parser = argparse.ArgumentParser(description='initialize horovod pods') 
parser.add_argument('--replicas', dest='replicas', required=True, help='number of MPI (Horovod) worker pods') 
parser.add_argument('--interactive-debugging-mode', dest='interactive_debugging_mode', default=False, help='put this pod to sleep for easier debugging') 
parser.add_argument('--is-head-node', dest='is_head_node', default=False, action='store_true', help='designates node as a head node') 
args = parser.parse_args()
args.replicas = int(args.replicas) 
if args.interactive_debugging_mode in [True, 'True', 'true']:
    args.interactive_debugging_mode = True
else:
    args.interactive_debugging_mode = False
    pass

def interactive_debugging_mode():
    print('starting in interactive debugging mode...')
    while True:
        print('sleeping 60 seconds...')
        sleep(60)
        pass 
    pass 

def write_ssh_aliases(n_replicas):
    'writes ~/.ssh/config'
    ssh_config_str = __get_ssh_config_str(n_replicas) 
    with open('/root/.ssh/config', 'w') as f:
        f.write(ssh_config_str) 
    pass

def __get_ssh_config_str(n_replicas):
    config_str = '' 
    for i in range(n_replicas):
        config_str += __add_ssh_host(i) 
        pass
    return config_str

def __add_ssh_host(idx):
    return f'''
Host horovod-{idx}
    HostName horovod-{idx}.horovod
    User root

'''

def wait_for_dns(replicas): 
    urls = [f'http://horovod-{idx}.horovod:22' for idx in range(replicas)] 
    while True:
        ready = True 
        for url in urls:
            try:
                _ = requests.get(url) 
            except Exception as e:
                if 'SSH' not in str(e): 
                    ## successful connection results in SSH rejecting non-secure request 
                    ready = False 
                    pass
                pass
            pass
        if ready:
            return ready 
        print('dns not ready, sleeping 30 seconds...') 
        sleep(30) 
    pass 

if __name__ == '__main__':
    ## write ssh host aliases
    ## without this, ssh cannot resolve full host names
    write_ssh_aliases(args.replicas)
    
    ## starting sshd
    os.system('service ssh start')
    
    if args.interactive_debugging_mode:
        interactive_debugging_mode()
        pass
    
    if args.is_head_node:
        wait_for_dns(args.replicas) 
        ## construct cmd 
        cmd = f'horovodrun -np {args.replicas} -H '
        for idx in range(args.replicas): 
            if idx > 0:
                cmd += ','
                pass 
            cmd += f'horovod-{idx}.horovod:1' 
            pass 
        cmd += ' xvfb-run python /app/src/python/ai/ai_runner.py'
        ## execute 
        run(cmd, os_system=True) 
        pass 

    ## check for master 
    pod_name = os.environ.get('POD_NAME') 
    print(f'POD_NAME: {pod_name}') 
    interactive_debugging_mode() 
    pass 
