from build.util import run

def helm_deploy_minio(root, conf): 
    ## install helm chart 
    cmd1 = f'helm repo add minio https://helm.min.io/'
    run(cmd1) 
    ## apply helm chart with values 
    cmd2 = f'helm upgrade minio minio/minio --install '+\
            f'-f {root}/src/helm/minio/values.yaml'
    run(cmd2) 
    pass 

