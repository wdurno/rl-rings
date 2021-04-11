import argparse
import os
from time import sleep

parser = argparse.ArgumentParser(description='initialize horovod pods') 
parser.add_argument('--replicas', dest='replicas', required=True, help='number of MPI (Horovod) worker pods') 
parser.add_argument('--interactive-debugging-mode', dest='interactive_debugging_mode', default=False, help='put this pod to sleep for easier debugging') 
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

if __name__ == '__main__':
    ## write ssh host aliases
    ## without this, ssh cannot resolve full host names
    write_ssh_aliases(args.replicas)
    
    ## starting sshd
    os.system('service ssh start')
    
    if args.interactive_debugging_mode:
        interactive_debugging_mode()
        pass
    
    ## check for master 
    pod_name = os.environ.get('POD_NAME') 
    print(f'POD_NAME: {pod_name}') 
    interactive_debugging_mode() 
    pass 
