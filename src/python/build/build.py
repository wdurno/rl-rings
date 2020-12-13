import argparse
import warnings 
from util import *

parser = argparse.ArgumentParser(description='Builds experimental infrastructure') 
parser.add_argument('--phase2', dest='phase2', action='store_true', help='build phase 2 (single node) architecture') 
parser.add_argument('--phase3', dest='phase3', action='store_true', help='build phase 3 (distributed) architecture') 
parser.add_argument('--clean-up', dest='clean_up', action='store_true', help='tear-down all infrastructure') 
parser.set_defaults(phase2=False, phase3=False, clean_up=False)
args = parser.parse_args() 

if (not args.phase2) and (not args.phase3) and (not args.clean_up):
    warnings.warn(WARNING+'No build args set. No further action will be taken.'+NC) 
    pass

if args.phase2:
    print(OKGREEN+'phase2 build initiated...'+NC)
    helm_deploy_build(name='build')
    copy_to_pod(pod_name='build') 
    build_phase_2_container(pod_name='build') 
    pass

if args.phase3:
    print(OKGREEN+'phase3 build initiated...'+NC)
    pass

if args.clean_up:
    print(OKGREEN+'tearing-down all infrastructure...'+NC) 
    tear_down() 
    pass 





