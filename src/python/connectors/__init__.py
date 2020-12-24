import os 
from connectors import minio_connector, postgres_connector, cassandra_connector 

## CONSTANTS 
mc = None
pc = None
cc = cassandra_connector.CassandraConnector('http://cassandra') 

## CONFIG 
if 'POSTGRES_SECRET' in os.environ:
    pc = postgres_connector.PostgresConnector('postgres'm os.environ['POSTGRES_SECRET']) 
    pass

if 'MINIO_ACCESSKEY' in os.environ and 'MINIO_SECRETKEY' in os.environ: 
    mc = minio_connector.MinIOConnector('minio:9000', {'accesskey': os.environ['MINIO_ACCESSKEY'], \
            'secretkey': os.environ['MINIO_SECRETKEY']})
    pass 

