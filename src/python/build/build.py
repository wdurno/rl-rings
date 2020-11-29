import os
import argparse

parser = argparse.ArgumentParser(description='Builds experimental infrastructure') 
parser.add_argument('--phase1', dest='phase1', action='store_true', help='build phase 1 (single node) architecture') 
parser.add_argument('--phase2', dest='phase2', action='store_true', help='build phase 2 (distributed) architecture') 
parser.set_defaults(phase1=False, phase2=False)

repo_dir = os.environ.get('repo-dir')
if repo_dir is None:
    raise ValueError('Please set `repo-dir` environment variable!') 



