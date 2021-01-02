import argparse
import warnings 
from util import *

parser = argparse.ArgumentParser(description='Builds experimental infrastructure') 
parser.add_argument('--phase-2', dest='phase_2', action='store_true', help='build phase 2 (single node) architecture') 
parser.add_argument('--phase-3', dest='phase_3', action='store_true', help='build phase 3 (distributed) architecture') 
parser.add_argument('--clean-up-compute', dest='clean_up_compute', action='store_true', help='tear-down compute, keep storage') 
parser.add_argument('--clean-up', dest='clean_up', action='store_true', help='tear-down all infrastructure') 
parser.add_argument('--no-build', dest='no_build', action='store_true', help='no Docker builds, just re-use')
parser.add_argument('--no-base-build', dest='no_base_build', action='store_true', help='do not build the base Docker image')
parser.add_argument('--keep-build-pod', dest='keep_build_pod', action='store_true', help='do not delete the build pod') 
parser.add_argument('--deploy-above-storage', dest='deploy_above_storage', action='store_true', help='deploy phase-3 non-storage') 
parser.set_defaults(phase2=False, phase3=False, clean_up=False)
args = parser.parse_args() 

if (not args.phase_2) and (not args.phase_3) and (not args.clean_up_compute) and (not args.clean_up):
    warnings.warn(WARNING+'No build args set. No further action will be taken.'+NC) 
    pass

if not args.no_build:
    print(OKGREEN+'running docker build...'+NC)  
    helm_deploy_build(name='build')
    copy_to_pod(pod_name='build')
    if not args.no_base_build:
        build_base_image(pod_name='build') 
    build_ai_image(pod_name='build')
    if not args.keep_build_pod: 
        helm_uninstall_build(name='build')
    pass 

print(OKGREEN+'refreshing ACR secret...'+NC)  
deploy_acr_secret()  

if args.phase_2:
    print(OKGREEN+'phase-2 build initiated...'+NC) 
    helm_deploy_phase_2_pod(name='phase-2') 
    pass

if args.phase_3:
    print(OKGREEN+'phase-3 build initiated...'+NC)
    terraform_deploy_phase_3() 
    helm_deploy_simulation_storage()
    helm_deploy_minio() 
    helm_deploy_postgres() 
    init_storage() ## blocks until complete
    helm_deploy_above_storage() 
    pass

if args.deploy_above_storage:
    helm_deploy_above_storage() 
    pass 

if args.clean_up_compute:
    print(OKGREEN+'tearing-down compute...'+NC) 
    tear_down_compute() 
    pass

if args.clean_up:
    print(OKGREEN+'tearing-down all infrastructure...'+NC) 
    tear_down() 
    pass 





