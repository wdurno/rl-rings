import os
import argparse
import warnings 

parser = argparse.ArgumentParser(description='Builds experimental infrastructure') 
parser.add_argument('--phase2', dest='phase2', action='store_true', help='build phase 2 (single node) architecture') 
parser.add_argument('--phase3', dest='phase3', action='store_true', help='build phase 3 (distributed) architecture') 
parser.set_defaults(phase2=False, phase3=False)

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

repo_dir = os.environ.get('repo-dir')
if repo_dir is None:
    raise ValueError(FAIL+'Please set `repo-dir` environment variable!'+NC) 

if (not phase2) and (not phase3):
    warnings.warn(WARNING+'Neither `phase2` nor `phase3` selected, no further build will occur'+NC) 
    pass

def run(cmd: str):
    'Execute a string as a blocking, exception-raising system call'
    exit_code = os.system(cmd) 
    if exit_code != 0:
        raise OSError(exit_code) 
    pass

if phase2:
    print(OKGREEN+'phase2 build initiated...'+NC)
    pass

if phase3:
    print(OKGREEN+'phase3 build initiated...'+NC)
    pass







