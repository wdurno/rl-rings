import argparse
import warnings 
from util import *

parser = argparse.ArgumentParser(description='Builds experimental infrastructure') 
parser.add_argument('--phase-2', dest='phase_2', action='store_true', help='build phase 2 (single node) architecture') 
parser.add_argument('--phase-3', dest='phase_3', action='store_true', help='build phase 3 (distributed) architecture') 
parser.add_argument('--clean-up-compute', dest='clean_up_compute', action='store_true', help='tear-down compute, keep storage') 
parser.add_argument('--clean-up', dest='clean_up', action='store_true', help='tear-down all infrastructure') 
parser.add_argument('--no-build', dest='no_build', action='store_true', help='no Docker builds, just re-use') 
parser.set_defaults(phase2=False, phase3=False, clean_up=False)
args = parser.parse_args() 

if (not args.phase_2) and (not args.phase_3) and (not args.clean_up_compute) and (not args.clean_up):
    warnings.warn(WARNING+'No build args set. No further action will be taken.'+NC) 
    pass

if args.phase_2:
    print(OKGREEN+'phase-2 build initiated...'+NC) 
    if not args.no_build: 
        helm_deploy_build(name='build')
        copy_to_pod(pod_name='build') 
        build_phase_2_container(pod_name='build') 
        helm_uninstall_build(name='build')
    deploy_acr_secret() 
    helm_deploy_phase_2_pod(name='phase-2') 
    pass

if args.phase_3:
    print(OKGREEN+'phase-3 build initiated...'+NC)
    pass

if args.clean_up_compute:
    print(OKGREEN+'tearing-down compute...'+NC) 
    tear_down_compute() 
    pass

if args.clean_up:
    print(OKGREEN+'tearing-down all infrastructure...'+NC) 
    tear_down() 
    pass 





