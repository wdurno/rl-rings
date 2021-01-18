## listen for incoming gradient shards, add them to the db

from flask import Flask, request, jsonify 
from waitress import serve 
from connectors import postgres_connector
import argparse 
import os 

parser = argparse.ArgumentParser(description='listen for gradient shards') 
parser.add_argument('--db-url', required=False, default='0.0.0.0', type=str) 
parser.add_argument('--db-password', required=False, default=None) 

app = Flask(__name__) 

@app.route('/', methods=['POST']) 
def add_grad():
    'post grad shard to local db for asynchronous integration'
    ## get grad shard data 
    data = request.get_data() 
    b64string = data.decode() ## assumes b64 bytes 
    ## write to db 
    local_pg.write_grad_shard(b64string) 
    print('grad written') 
    return jsonify('success') 

if __name__ == '__main__': 
    ## config 
    args = parser.parse_args() 
    db_password = args.db_password
    if db_password is None:
        ## get from env
        db_password = os.environ['POSTGRES_SECRET']
    db_password = db_password.replace('\n', '')
    ## init DB connector 
    local_pg = postgres_connector.PostgresConnector(args.db_url, db_password)
    ## serve 
    serve(app, host='0.0.0.0', port=5000) 
