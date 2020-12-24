import os 
import jinja2
from subprocess import Popen, PIPE

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

def run(cmd: str, stdin: str=None):
    'Execute a string as a blocking, exception-raising system call'
    ## verify assumptions 
    if type(cmd) != str:
        raise ValueError('`cmd` must be a string!')
    ## execute 
    print(OKCYAN+cmd+NC)
    exit_code = os.system(cmd)
    if stdin is None: 
        ## no stdin 
        proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        exit_code = proc.wait() 
        stdout = proc.stdout.read().decode() 
        stderr = proc.stderr.read().decode() 
    else:
        ## apply stdin 
        if type(stdin) not in [str, bytes]:
            raise ValueError('STDIN must be str or bytes!')
        if type(stdin) == str:
            ## convert to bytes
            stdin = stdin.encode() 
        proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, stdin=PIPE) 
        stdout, stderr = proc.communicate(stdin)
        stdout = stdout.decode() 
        stderr = stderr.decode() 
        exit_code = proc.returncode 
    if exit_code != 0:
        print(FAIL+'STDOUT: '+stdout+NC) 
        print(FAIL+'STDERR: '+stderr+NC) 
        raise OSError(exit_code)
    pass

def tear_down(): 
    'tear-down all infrastructure'
    cmd = f'cd {repo_dir}/src/terraform/state && '+\
        'bash terraform-destroy.sh'
    run(cmd)
    pass 

def tear_down_compute():
    'leave storage intact'
    cmd1 = f'rm {repo_dir}/src/terraform/state/k8s.tf '+\
            '{repo_dir}/src/terraform/state/ephemeral_pool.tf '+\
            '{repo_dir}/src/terraform/state/storage_pool.tf' 
    try:
        run(cmd1) 
    except:
        ## it's fine if file is already deleted
        pass
    cmd2 = f'cd {repo_dir}/src/terraform/state && '+\
        '. terraform-apply.sh'
    run(cmd2) 
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
    ## `repo_dir` not referenced, because local `repo_dir` is different from build env `repo_dir`
    ## so, it is hard-coded as `/build/rl-hypothesis-2`
    cmd = f'kubectl exec -it {pod_name} -- sh /build/rl-hypothesis-2/phase-2-single-node/build.sh'
    run(cmd) 
    pass

def helm_deploy_phase_2_pod(name='phase-2'): 
    'deploys phase 2 pod for single-node AI processing'
    cmd = f'helm upgrade {name} {repo_dir}/src/helm/phase-2 --install '+\
        f'--set name={name} '+\
        f'--set docker_server=$(cat {repo_dir}/secret/acr/server)'
    run(cmd) 
    pass

def deploy_acr_secret(): 
    'Refreshes ACR secret'
    cmd1 = f'kubectl delete secret acr-creds'
    try:
        run(cmd1) 
    except:
        ## if secret doesn't exist yet, just create a new one 
        pass
    cmd2 = 'kubectl create secret docker-registry acr-creds '+\
        f'--docker-server=$(cat {repo_dir}/secret/acr/server) '+\
        '--docker-username=00000000-0000-0000-0000-000000000000 '+\
        f'--docker-password=$(cat {repo_dir}/secret/acr/token)'
    run(cmd2) 
    pass

def terraform_deploy_phase_3(): 
    'sets up necessary compute infrastructure for phase-3'
    ## add required infra spec 
    cmd1 = f'cp {repo_dir}/src/terraform/templates/phase-3/* {repo_dir}/src/terraform/state'
    run(cmd1) 
    ## apply 
    cmd2 = f'cd {repo_dir}/src/terraform/state && '+\
            '. terraform-apply.sh'
    run(cmd2) 
    pass 

def helm_deploy_simulation_storage(): 
    cmd = f'helm upgrade simulation-storage {repo_dir}/src/helm/simulation-storage --install'
    run(cmd) 
    pass 

def helm_deploy_minio(): 
    ## install helm chart 
    cmd1 = f'helm repo add minio https://helm.min.io/'
    run(cmd1) 
    ## apply helm chart with values 
    cmd2 = f'helm upgrade minio minio/minio --install '+\
            f'-f {repo_dir}/src/helm/minio/values.yaml'
    run(cmd2) 
    pass 

def helm_deploy_postgres():
    cmd = f'helm upgrade postgres {repo_dir}/src/helm/postgres --install'
    run(cmd) 
    pass 

def init_storage(): 
    ## get job template 
    with open(f'{repo_dir}/src/k8s/init-storage-job.yaml', 'r') as f:
        job_template = jinja2.Template(f.read()) 
    ## get variable for template 
    with open(f'{repo_dir}/secret/acr/server') as f: 
        docker_server = f.read() 
    ## populate 
    job_yaml = job_template.render(docker_server=docker_server) 
    ## apply 
    cmd = 'kubectl apply -f -'
    stdin = job_yaml.encode() 
    run(cmd, stdin=stdin) 
    pass 

