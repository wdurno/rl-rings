import os 
import argparse 
from time import sleep 
from connectors import minio_connector, postgres_connector, cassandra_connector

parser = argparse.ArgumentParser(description='Initialize storage services') 
parser.add_argument('--minio', dest='minio', action='store_true', help='initialize minio') 
parser.add_argument('--postgres', dest='postgres', action='store_true', help='initialize postgres') 
parser.add_argument('--cassandra', dest='cassandra', action='store_true', help='initialize cassandra') 
args = parser.parse_args() 

## contstants 
MINIO_ACCESSKEY = os.environ['MINIO_ACCESSKEY'] 
MINIO_SECRETKEY = os.envrion['MINIO_SECRETKEY'] 
POSTGRES_SECRET = os.environ['POSTGRES_SECRET'] 

def continue_attempting(func, wait_time=10): 
    continue_trying = True 
    while continue_trying: 
        try:
            func() 
            continue_trying = False 
        except Exception as e:
            print('exception encountered, waiting '+str(wait_time)+' seconds and reattempting...') 
            print(e)
            wait(wait_time) 
    pass

def init_minio():
    mc = minio_connector.MinIOConnector('minio:9000', {'accesskey': MINIO_ACCESSKEY, 'secretkey': MINIO_SECRETKEY}) 
    mc.init_storage() 
    pass 

def init_postgres(): 
    pc = postgres_connector.PostgresConnector('postgres', POSTGRES_SECRET)  
    pass 

def init_cassandra():
    cc = cassandra_connector.CassandraConnector('http://cassandra') 
    cc.init_storage() 
    pass 

if args.minio:
    continue_attempting(init_minio) 
    pass

if args.postgres:
    continue_attempting(init_postgres) 
    pass

if args.cassandra:
    continue_attempting(init_cassandra) 
    pass 

