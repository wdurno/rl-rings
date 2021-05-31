from build.util import run
import os
import base64 

def helm_deploy_postgres(root, conf):
    ## update local secret 
    postgres_secret = base64.urlsafe_b64encode(os.urandom(16)).decode() 
    with open(f'{root}/secret/postgres/postgres-secret', 'w') as f:
        f.write(postgres_secret) 
    cmd1 = f'kubectl delete secret postgres'
    cmd2 = f'kubectl create secret generic postgres --from-file={root}/secret/postgres/postgres-secret'
    try: 
        run(cmd1)
    except:
        print('failed to delete remote postgres secret, probably because it does not exist')
        pass 
    run(cmd2) 
    ## deploy 
    cmd3 = f'helm upgrade postgres {root}/src/helm/postgres --install'
    run(cmd3) 
    pass 

