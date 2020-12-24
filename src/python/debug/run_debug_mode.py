import os
import jinja2
from subprocess import Popen, PIPE

## TODO this is duplicated code in this file. See python/build/util.py. Refactor required. 
repo_dir = os.environ['repo-dir'] 

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

def init_storage():
    ## get job template
    with open(f'{repo_dir}/src/k8s/debug-pod.yaml', 'r') as f:
        pod_template = jinja2.Template(f.read())
    ## get variable for template
    with open(f'{repo_dir}/secret/acr/server') as f:
        docker_server = f.read()
    ## populate
    pod_yaml = pod_template.render(docker_server=docker_server)
    ## apply
    cmd = 'kubectl apply -f -'
    stdin = pod_yaml.encode()
    run(cmd, stdin=stdin)
    pass

