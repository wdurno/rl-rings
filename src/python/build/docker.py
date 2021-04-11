from build.util import run
from time import sleep 

def docker_build(root, conf, keep_docker_build_env=False): 
    '''
    Builds necessary docker images. 
    Works by 
    1. deploying a build env container, 
    2. copying build dependencies into the container, 
    3. building the image, 
    4. uploading, 
    5. then tearing-down the build env. 
    '''
    __deploy_docker_build_env(root, conf) 
    __build(root, conf) 
    if not keep_docker_build_env:
        __tear_down_docker_build_env(root, conf) 
        pass
    pass 

def __deploy_docker_build_env(root, conf, blocking=True): 
    'deploys build env helm chart'
    ## deploy build 
    name = 'build'
    cmd1 = f'helm upgrade {name} {root}/src/helm/build/ --install '+\
        f'--set name={name} ' 
    run(cmd1) 
    if blocking:
        ## wait until deployed 
        cmd2 = f'kubectl wait --for=condition=ready pod -l name={name}'
        run(cmd2) 
        ## docker daemon needs a little more time 
        sleep(3) 
    pass 

def __build(root, conf): 
    'runs a remote docker build'
    ## load secrets 
    cmd1 = f'cat {root}/secret/acr/server' 
    cmd2 = f'cat {root}/secret/acr/token' 
    acr_server = run(cmd1, return_stdout=True) 
    acr_token = run(cmd2, return_stdout=True) 
    ## setup build environment 
    cmd3 = f'kubectl exec build -- mkdir -p /build' 
    cmd4 = f'kubectl cp {root}/docker build:/build/docker && kubectl cp {root}/src build:/build/docker/src' 
    run(cmd3) 
    run(cmd4) 
    ## build 
    image_name = acr_server + '/' + conf['image_name']
    acr_name = conf['terraform_prefix'] + 'acr'
    cmd5 = f'kubectl exec -it build -- sh -c "cd /build/docker && docker build -t {image_name} ."' 
    run(cmd5, os_system=True) 
    ## push 
    cmd6 = f'kubectl exec -it build -- docker login {acr_server} --username {acr_name} --password {acr_token}' 
    cmd7 = f'kubectl exec -it build -- docker push {image_name}' 
    run(cmd6) 
    run(cmd7, os_system=True) 
    pass

def __tear_down_docker_build_env(root, conf): 
    'tears-down build env helm chart'
    name = 'build' 
    cmd = f'helm uninstall {name}' 
    run(cmd) 
    pass 

