from build.util import run

def cassandra_deploy(root, conf): 
    'deploys horovod'
    ## config 
    cassandra_instances = int(conf['cassandra_instances']) 
    ## helm deploy 
    cmd1 = f'helm upgrade --install cassandra-ring {root}/src/helm/cassandra-ring '+\
            f'--set cassandra.replicas={cassandra_instances}'
    run(cmd1, os_system=True) 
    pass

