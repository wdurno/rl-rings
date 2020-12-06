import os
import argparse
import warnings 

parser = argparse.ArgumentParser(description='Builds experimental infrastructure') 
parser.add_argument('--phase2', dest='phase2', action='store_true', help='build phase 2 (single node) architecture') 
parser.add_argument('--phase3', dest='phase3', action='store_true', help='build phase 3 (distributed) architecture') 
parser.add_argument('--clean-up', dest='clean_up', action='store_true', help='tear-down all infrastructure') 
parser.set_defaults(phase2=False, phase3=False, clean_up=False)
args = parser.parse_args() 

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

repo_dir = os.environ.get('repo_dir')
if repo_dir is None:
    raise ValueError(FAIL+'Please set `repo-dir` environment variable!'+NC) 

if (not args.phase2) and (not args.phase3) and (not args.clean_up):
    warnings.warn(WARNING+'No build args set. No further action will be taken.'+NC) 
    pass

def run(cmd: str):
    'Execute a string as a blocking, exception-raising system call'
    if type(cmd) != str: 
        raise ValueError('`cmd` must be a string!')
    print(OKGREEN+cmd+NC)
    exit_code = os.system(cmd) 
    if exit_code != 0:
        raise OSError(exit_code) 
    pass

if args.phase2:
    print(OKGREEN+'phase2 build initiated...'+NC)
    pass

if args.phase3:
    print(OKGREEN+'phase3 build initiated...'+NC)
    pass

if args.clean_up:
    print(OKGREEN+'tearing-down all infrastructure...'+NC) 
    cmd = f'cd {repo_dir}/src/terraform/state && '+\
        'bash terraform-destroy.sh' 
    run(cmd) 
    pass 





