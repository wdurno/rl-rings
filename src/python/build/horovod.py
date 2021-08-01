from build.util import run
from time import sleep 

def deploy_horovod(root, conf): 
    'deploys horovod'
    interactive_debugging_mode = conf['interactive_debugging_mode']
    ## build image name  
    cmd1 = f'cat {root}/secret/acr/server' 
    acr_server = run(cmd1, return_stdout=True) 
    image_name = acr_server + '/ai' + conf['image_tag']
    horovod_instances = int(conf['horovod_instances']) 
    ## helm deploy 
    cmd2 = f'helm upgrade --install horovod-ring {root}/src/helm/horovod-ring '+\
            f'--set image={image_name} '+\
            f'--set interactive_debugging_mode={interactive_debugging_mode} '+\
            f'--set replicas={horovod_instances}'
    run(cmd2, os_system=True) 
    pass

def update_horovod_worker_src(root, conf):
    'updates horovod worker src directory'
    horovod_instances = int(conf['horovod_instances']) 
    for worker_idx in range(horovod_instances): 
        ## delete remote src 
        cmd1 = f'kubectl exec -it horovod-{worker_idx} -- rm -rf /app/src'
        run(cmd1, os_system=True) 
        ## copy local src to remote 
        cmd2 = f'kubectl cp {root}/src horovod-{worker_idx}:/app/src'
        run(cmd2, os_system=True) 
        pass 
    pass
