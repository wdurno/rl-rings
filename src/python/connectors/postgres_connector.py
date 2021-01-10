
from connectors.storageABC import __StorageABC
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT 
import uuid 
from datetime import datetime, timedelta  
import pandas as pd 
import os 

class PostgresConnector(__StorageABC): 
    'Manipulate a postgres instance'

    def __init__(self, url, secret): 
        'init storage interface instance' 
        super().__init__(url, secret) 
        self.connection = None 
        pass 

    def __get_connection(self):
        'Get live connection to storage.'
        if self.connection is None: 
            self.connection = psycopg2.connect(dbname="structured", user="postgres", host=self.url, port="5432", password=self.secret)
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) 
        return self.connection 

    def __exec(self, sql, debug=False): 
        'Execute sql. Connectivity abstracted-away.'
        if debug:
            print('POSTGRES EXEC: '+str(sql)) 
        ## define attempt mechanism 
        def run_sql(sql):
            connection = self.__get_connection() 
            cur = connection.cursor() 
            cur.execute(sql) 
            if cur.description is None: 
                ## no rows to return 
                return [] 
            rows = cur.fetchall() 
            return rows 
        try:
            ## first attempt, assuming connectivity 
            return run_sql(sql) 
        except psycopg2.OperationalError: 
            ## assuming connectivity error...
            ## attempting reconnection and trying again... 
            self.connection = None 
            return run_sql(sql) 
        pass 

    def close_connection(self): 
        'Close any existing storage connection'
        if self.connection is not None: 
            self.connection.close()
            self.connection = None 
        pass 

    #### DISTRIBUTED STORAGE METHODS 

    def init_storage(self): 
        '''
        Initialize storage instance for the entire cluster. 
        You should only need to run this once. 
        '''
        sql1 = 'CREATE DATABASE structured;'
        sql2 = '''
               CREATE TABLE metrics(game CHAR[20], 
                   model INT4,
                   reward FLOAT4,
                   etc FLOAT4);
               ''' ## TODO finish data model
        sql3 = 'CREATE TABLE transitions(transition_id UUID PRIMARY KEY);'
        sql4 = 'CREATE TABLE grad_ids(grad_id UUID PRIMARY KEY, timestamp TIMESTAMP);'
        sql5 = 'CREATE TABLE latest_model(model_id INT4 PRIMARY KEY, path TEXT);'
        sql6 = "INSERT INTO latest_model VALUES (0, '');" 
        sql7 = 'CREATE TABLE parameter_server_state(last_model_publish_time TIMESTAMP, last_grad_time TIMESTAMP);'
        sql8 = 'CREATE TABLE parameter_sherver_shards(shard_id UUI PRIMARY KEY, shard_index INT4, timestamp TIMESTAMP);'
        ## init DB requires special connection 
        self.connection = psycopg2.connect(user='postgres', host=self.url, port='5432', password=self.secret)
        self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) 
        self.__exec(sql1, debug=True)
        self.close_connection() 
        ## normal requests with normal connections 
        self.__exec(sql2, debug=True) 
        self.__exec(sql3, debug=True) 
        self.__exec(sql4, debug=True) 
        self.__exec(sql5, debug=True) 
        self.__exec(sql6, debug=True) 
        ## self.__exec(sql7, debug=True) # deprecated  
        self.__exec(sql8, debug=True)
        pass

    def write_transition_id(self, _uuid):
        'write a transition uuid to postgres'
        sql = f"INSERT INTO transitions VALUES ('{_uuid}')" 
        self.__exec(sql) 
        pass
    
    def get_total_transitions(self): 
        sql = 'SELECT COUNT(*) FROM transitions;'
        rows = self.__exec(sql) 
        return rows[0][0] 
    
    def sample_transition_ids(self, expected_transitions: int):
        total_transitions = self.get_total_transitions() 
        if total_transitions == 0:
            return [] 
        prob = min(100, 100*expected_transitions/total_transitions) 
        sql = f'SELECT transition_id FROM transitions TABLESAMPLE BERNOULLI({prob});'
        uuids = [r[0] for r in self.__exec(sql)] 
        return uuids 

    def set_model_path(self, path: str): 
        'update model details for general consumption'
        ## execute update 
        sql1 = f"UPDATE latest_model SET model_id = model_id+1, path = '{path}';" 
        self.__exec(sql1) 
        ## verify legitimacy 
        sql2 = 'SELECT model_id FROM latest_model'
        model_id_row_list = self.__exec(sql2) 
        if len(model_id_row_list) != 1: 
            raise Exception('Postgres table `latest_model` should have exactly 1 row!') 
        ## return model_id
        return model_id_row_list[0][0] 

    def get_latest_model_path(self): 
        'get latest MinIO path to model, returning (model_id, path)'
        sql = 'SELECT model_id, path FROM latest_model;'
        row_list = self.__exec(sql) 
        model_id = row_list[0][0] 
        path = row_list[0][1] 
        return model_id, path 

    def get_registered_grad_id(self): 
        'register a gradient uuid with postgres and return it'
        ## define 
        grad_id = uuid.uuid1() 
        timestamp = datetime.now()  
        ## register 
        sql = f"INSERT INTO grad_ids VALUES ('{grad_id}', '{timestamp}');" 
        self.__exec(sql) 
        return grad_id 

    def get_grad_ids_after_timestamp(self, timestamp): 
        'returns dataframe of columns (grad_id, ts) where ts > timestamp'
        ## get data 
        sql = f"SELECT grad_id, timestamp FROM grad_ids WHERE timestamp > '{timestamp}';"
        rows = self.__exec(sql) 
        ## format as dataframe 
        grad_ids = [row[0] for row in rows] 
        timestamps = [row[1] for row in rows] 
        return pd.DataFrame({'grad_id': grad_ids, 'timestamp': timestamps}) 

    def get_parameter_server_state(self): 
        '''
        Gets state, initializing if necessary.
        Returns (last_model_publish_time, last_grad_time) 
        '''
        ## get rows 
        sql = 'SELECT last_model_publish_time, last_grad_time FROM parameter_server_state;'
        rows = self.__exec(sql) 
        if len(rows) < 1:
            ## need to init 
            now = datetime.now() 
            sql = f"INSERT INTO parameter_server_state VALUES ('{now}', '{now}');"
            self.__exec(sql) 
            return now, now
        ## no need to init 
        return rows[0][0], rows[0][1] 

    def update_parameter_server_state(self, last_model_publish_time=None, last_grad_time=None): 
        if last_model_publish_time is not None:
            sql1 = f"UPDATE parameter_server_state SET last_model_publish_time='{last_model_publish_time}';"
            self.__exec(sql1) 
            pass
        if last_grad_time is not None:
            sql2 = f"UPDATE parameter_server_state SET last_grad_time='{last_grad_time}';"
            self.__exec(sql2) 
            pass 
        pass 

    def register_parameter_server_shard(self, _uuid, shard_index): 
        'after writing shard to cassandra, register uuid here' 
        timestemp = datetime.now() 
        sql = f"INSERT INTO parameter_server_shards VALUES ('{_uuid}', {shard_index}, 'timestamp');" 
        self.__exec(sql)  
        pass 

    def get_latest_parameter_server_shard_uuid(self, shard_index): 
        'returns a uuid str or None'
        sql = f'''
        SELECT shard_id WHERE timestamp=max_timestamp FROM parameter_server_shards IN
        ( 
            SELECT shard_id, timestamp, MAX(timestamp) AS max_timestamp
            FROM parameter_server_shards
            WHERE shard_id={shard_index} 
        )x;
        '''
        rows = self.__exec(sql) 
        if len(rows) > 1: 
            raise ValueError('Bad sql returns multiple rows!\n'+sql)  
        if len(rows) == 1:
            return str(rows[0][0]) 
        return None 

    def get_all_latest_parameter_server_shard_uuids(self): 
        sql = '''
        SELECT shard_id, shard_index FROM 
        (
            SELECT shard_id, MAX(timestamp) 
            FROM parameter_server_shards 
            GROUP BY shard_index 
        )x
        ORDER BY shard_index; 
        '''
        row_list = self.__exec(sql) 
        return [(str(row[0]), int(row[1])) for row in row_list]

    #### PARAMETER SERVER SHARD LOCAL DB 

    def init_grad_shard_storage(self): 
        sql1 = 'CREATE DATABASE structured;' 
        sql2 = 'CREATE TABLE timestamped_b64_store(b64string TEXT, timestamp TIMESTAMP);'
        self.connection = psycopg2.connect(user='postgres', host=self.url, port='5432', password=self.secret) 
        self.__exec(sql1) 
        self.close_connection()  
        self.__exec(sql2) 
        pass 

    def write_grad_shard(self, b64string): 
        timestamp = datetime.now() 
        sql1 = f"INSERT INTO timestamped_b64_store VALUES ('{b64string}', '{timestamp}');" 
        self.__exec(sql1) 
        pass 

    def read_grad_shard_after_timestamp(self, timestamp): 
        sql = f"SELECT b64string FROM timestamped_b64_store WHERE timestamp >= '{timestamp}';" 
        row_list = self.__exec(sql) 
        return [b64_row[0] for b64_row in row_list] 

    def delete_grad_shards_before_timestamp(self, timestamp): 
        sql = f"SELECT FROM timestamped_b64_store WHERE timestamp < '{timestamp}';" 
        self.__exec(sql) 
        pass 
    pass
