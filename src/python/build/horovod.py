from build.util import run
from time import sleep 

def deploy_horovod(root, conf): 
    'deploys horovod'
    interactive_debugging_mode = conf['interactive_debugging_mode']
    ## build image name  
    cmd1 = f'cat {root}/secret/acr/server' 
    acr_server = run(cmd1, return_stdout=True) 
    image_name = acr_server + '/' + conf['image_name']
    ## helm deploy 
    cmd2 = f'helm upgrade --install horovod {root}/src/helm/horovod '+\
            f'--set image={image_name} '+\
            f'--set interactive_debugging_mode={interactive_debugging_mode}'
    run(cmd2, os_system=True) 
    pass


