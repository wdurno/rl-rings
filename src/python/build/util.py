import os 
import jinja2
import random
import string
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
ai_image_tag = os.environ.get('rl_hypothesis_2_ai_image_tag')
if ai_image_tag is None: 
    raise ValueError(FAIL+'Please set `rl_hypothesis_2_ai_image_tag` environment variable!'+NC)
n_shards = os.environ.get('rl_hypothesis_2_n_parameter_server_shards') 
if n_shards is None: 
    raise ValueError(FAIL+'Please set `rl_hypothesis_2_n_parameter_server_shards` environment variable!'+NC) 
n_shards = int(n_shards) 

def run(cmd: str, stdin: str=None, os_system: bool=False):
    'Execute a string as a blocking, exception-raising system call'
    ## verify assumptions 
    if type(cmd) != str:
        raise ValueError('`cmd` must be a string!')
    ## execute 
    print(OKCYAN+cmd+NC)
    if stdin is None: 
        ## no stdin 
        if not os_system:
            proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            exit_code = proc.wait() 
            stdout = proc.stdout.read().decode() 
            stderr = proc.stderr.read().decode() 
        else:
            exit_code = os.system(cmd)
            stdout = 'not captured'
            stderr = 'not captured'
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
    run(cmd, os_system=True) 
    pass 

def tear_down_compute():
    'leave storage intact'
    cmd1 = f'rm {repo_dir}/src/terraform/state/k8s.tf '+\
            f'{repo_dir}/src/terraform/state/ephemeral_pool.tf '+\
            f'{repo_dir}/src/terraform/state/storage_pool.tf' 
    try:
        run(cmd1) 
    except:
        ## it's fine if file is already deleted
        pass
    cmd2 = f'cd {repo_dir}/src/terraform/state && '+\
        '. terraform-apply.sh'
    run(cmd2, os_system=True) 
    pass 

def helm_deploy_build(name='build-1', blocking=True): 
    'deploy a DinD pod'
    ## deploy build 
    cmd1 = f'helm upgrade {name} {repo_dir}/src/helm/build/ --install '+\
        f'--set name={name} ' 
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
    ## delete any old content 
    cmd1 = f'kubectl exec -it {pod_name} -- rm -rf {dst}/*'
    cmd2 = f'kubectl exec -it {pod_name} -- rm /root/rl-hypothesis-2-config.sh'
    try: 
        run(cmd1, os_system=True) 
        run(cmd2, os_system=True) 
    except:
        ## no need to delete that which does not exist 
        pass 
    ## copy content into pod  
    cmd3 = f'kubectl cp {src} {pod_name}:{dst}' 
    cmd4 = f'kubectl cp ~/rl-hypothesis-2-config.sh {pod_name}:/root/rl-hypothesis-2-config.sh'
    run(cmd3) 
    run(cmd4)  
    pass 

def build_base_image(pod_name='build-1'):
    'execute base-image build on remote pod'
    ## `repo_dir` not referenced, because local `repo_dir` is different from build env `repo_dir`
    ## so, it is hard-coded as `/build/rl-hypothesis-2`
    cmd = f'kubectl exec -it {pod_name} -- sh /build/rl-hypothesis-2/build/base-image/build.sh'
    run(cmd, os_system=True) 
    pass

def build_ai_image(pod_name='build-1'):
    'execute ai image build on remote pod'
    ## `repo_dir` not referenced, because local `repo_dir` is different from build env `repo_dir`
    ## so, it is hard-coded as `/build/rl-hypothesis-2`
    cmd = f'kubectl exec -it {pod_name} -- sh /build/rl-hypothesis-2/build/ai/build.sh'
    run(cmd, os_system=True)
    pass

def helm_deploy_phase_2_pod(name='phase-2'): 
    'deploys phase 2 pod for single-node AI processing'
    cmd = f'helm upgrade {name} {repo_dir}/src/helm/phase-2 --install '+\
        f'--set name={name} '+\
        f'--set docker_server=$(cat {repo_dir}/secret/acr/server) '+\
        f'--set ai_image_tag="{ai_image_tag}"'
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
        '--docker-username=RlHypothesis2AzureContainerRegsitry1 '+\
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
    run(cmd2, os_system=True) 
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
    ## init secret 
    cmd1 = f'. {repo_dir}/secret/postgres/make_postgres_secret.sh'
    cmd2 = f'kubectl delete secret postgres'
    cmd3 = f'kubectl create secret generic postgres --from-file={repo_dir}/secret/postgres/postgres-secret'
    run(cmd1) 
    try: 
        run(cmd2)
    except:
        ## no need to delete inexistent secret 
        pass 
    run(cmd3) 
    ## deploy 
    cmd4 = f'helm upgrade postgres {repo_dir}/src/helm/postgres --install'
    run(cmd4) 
    pass 

def random_str(n_char=5): 
    letters = string.ascii_lowercase 
    return ''.join(random.choice(letters) for i in range(n_char))

def init_storage(): 
    ## get job template 
    with open(f'{repo_dir}/src/k8s/init-storage-job.yaml', 'r') as f:
        job_template = jinja2.Template(f.read()) 
    ## get variable for template 
    with open(f'{repo_dir}/secret/acr/server') as f: 
        docker_server = f.read() 
    ## generate random id 
    rand_id = random_str()  
    ## populate 
    job_yaml = job_template.render(docker_server=docker_server, rand_id=rand_id, ai_image_tag=ai_image_tag) 
    ## apply 
    cmd1 = 'kubectl apply -f -'
    stdin = job_yaml.encode() 
    run(cmd1, stdin=stdin) 
    ## block until complete 
    cmd2 = f'kubectl wait --timeout=-1s --for=condition=complete job/init-storage-{rand_id}' 
    run(cmd2) 
    pass 

def helm_deploy_simulation(): 
    cmd = f'helm upgrade simulation {repo_dir}/src/helm/simulation --install '+\
            f'--set docker_server=$(cat {repo_dir}/secret/acr/server) '+\
            f'--set ai_image_tag="{ai_image_tag}"'
    run(cmd) 
    pass 

def helm_deploy_parameter_server_shard(shard_index:int): 
    shard_index = str(shard_index)
    cmd = f'helm upgrade parameter-server-shard-{shard_index} {repo_dir}/src/helm/parameter-server-shard --install '+\
            f'--set name="parameter-server-shard-{shard_index}" '+\
            f'--set shard_index="{shard_index}" '+\
            f'--set docker_server=$(cat {repo_dir}/secret/acr/server) '+\
            f'--set ai_image_tag="{ai_image_tag}"'
    run(cmd) 
    pass 

def helm_deploy_parameter_server_shards(): 
    for idx in range(n_shards): 
        helm_deploy_parameter_server_shard(idx) 
    pass 

def helm_deploy_gradient_calculation(): 
    cmd = f'helm upgrade gradient-calculation {repo_dir}/src/helm/gradient-calculation --install '+\
            f'--set docker_server=$(cat {repo_dir}/secret/acr/server) '+\
            f'--set ai_image_tag="{ai_image_tag}" '+\
            f'--set total_gradient_shards="{n_shards}"'
    run(cmd) 
    pass

def helm_deploy_parameter_shard_combiner():
    cmd = f'helm upgrade parameter-shard-combiner {repo_dir}/src/helm/parameter-shard-combiner --install '+\
            f'--set docker_server=$(cat {repo_dir}/secret/acr/server) '+\
            f'--set ai_image_tag="{ai_image_tag}" '+\
            f'--set total_gradient_shards="{n_shards}"'
    run(cmd) 
    pass

def helm_deploy_above_storage():
    helm_deploy_simulation() 
    helm_deploy_parameter_server_shards() 
    helm_deploy_parameter_shard_combiner() 
    helm_deploy_gradient_calculation() 
    pass 
