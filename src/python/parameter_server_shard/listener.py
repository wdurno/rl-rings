## listen for incoming gradient shards, add them to the db

from flask import Flask, request, jsonify 
from waitress import serve 
from connectors import postgres_connector
import argparse 
import os 

parser = argparse.ArgumentParser(description='listen for gradient shards') 
parser.add_argument('--db-url', required=False, default='0.0.0.0', type=str) 
parser.add_argument('--db-password', required=False, default=None) 

app = Flast(__name__) 

@app.route('/', methods=['POST']) 
def add_grad():
    'post grad shard to local db for asynchronous integration'
    ## constants 
    args = parser.parse_args() 
    db_password = args.db_password 
    if db_password is None: 
        ## get from env 
        db_password = os.environ['POSTGRES_SECRET']
    db_password = db_password.replace('\n', '')
    ## get grad shard data 
    data = request.get_data() 
    b64string = data.encode() ## assumes b64 bytes 
    ## write to db 
    local_pg = postgres_connector(args.db_url, args.db_password.replace('\n', '')) 
    local_pg.write_grad_shard(b64string) 
    local_pg.close_connection() 
    return jsonify('success') 

if __name__ == '__main__': 
    args = parser.parse_args() 
    serve(app, host='0.0.0.0', port=5000) 
