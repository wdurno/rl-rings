import os 

## constants 
## text color constants
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
NC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

## path to main repo dir 
repo_dir = os.environ.get('repo_dir')
if repo_dir is None:
    raise ValueError(FAIL+'Please set `repo-dir` environment variable!'+NC) 

def run(cmd: str):
    'Execute a string as a blocking, exception-raising system call'
    if type(cmd) != str:
        raise ValueError('`cmd` must be a string!')
    print(OKCYAN+cmd+NC)
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise OSError(exit_code)
    pass

def tear_down(): 
    'tear-down all infrastructure'
    cmd = f'cd {repo_dir}/src/terraform/state && '+\
        'bash terraform-destroy.sh'
    run(cmd)
    pass 

def helm_deploy_build(name='build-1', blocking=True): 
    'deploy a DinD pod'
    ## deploy build 
    cmd1 = f'helm upgrade {name} {repo_dir}/src/helm/build/ --install '+\
        f'--set name={name}'
    run(cmd1) 
    if blocking:
        ## wait until deployed 
        cmd2 = f'kubectl wait --for=condition=ready pod -l name={name}'
        run(cmd2) 
    pass 

def helm_uninstall_build(name='build-1'): 
    'tear-down a DinD pod'
    cmd = f'helm uninstall {name}' 
    run(cmd) 
    pass 

def copy_to_pod(pod_name='build-1', src=repo_dir, dst='/build'):
    'copy from local to pod remote'
    cmd = f'kubectl cp {src} {pod_name}:{dst}' 
    run(cmd) 
    pass 

def build_phase_2_container(pod_name='build-1'): 
    'executes build on remote pod'
    cmd = f'kubectl exec -it {pod_name} -- sh /build/rl-hypothesis-2/phase-2-single-node/build.sh'
    run(cmd) 
    pass

