
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
               CREATE TABLE sim_metrics(
                   game TEXT, 
                   model TEXT,
                   reward FLOAT4,
                   frames INT4);
               ''' 
        sql3 = 'CREATE TABLE transitions(transition_id UUID PRIMARY KEY, manually_generated BOOLEAN);'
        sql4 = 'CREATE TABLE latest_model(model_id INT4 PRIMARY KEY, path TEXT);'
        sql5 = "INSERT INTO latest_model VALUES (0, '');" 
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
        pass

    def write_sim_metrics(self, game: str, model: str, reward: float, frames: int):
        'write simulation metrics to postgres'
        sql = f"INSERT INTO sim_metrics VALUES ('{game}', '{model}', {reward}, {frames});"
        self.__exec(sql)
        pass 

    def get_metrics(self): 
        'return metrics as a dict of dataframes' 
        ## get sim metrics 
        sql1 = 'SELECT * FROM sim_metrics;'
        rows = self.__exec(sql1) 
        ## build sim metrics df 
        game = [row[0] for row in rows] 
        model = [row[1] for row in rows] 
        reward = [row[2] for row in rows] 
        frames = [row[3] for row in rows] 
        sim_metrics =  pd.DataFrame({'game': game, 'model': model, 'reward': reward, 'frames': frames}) 
        ## get and build grad metrics 
        sql2 = 'SELECT * FROM grad_metrics;' 
        rows = self.__exec(sql2) 
        game = [row[0] for row in rows] 
        model = [row[1] for row in rows] 
        loss = [row[2] for row in rows] 
        q_pred = [row[3] for row in rows] 
        grad_metrics = pd.DataFrame({'game': game, 'model': model, 'loss': loss, 'q_pred': q_pred}) 
        ## combine DFs 
        data_frames = {'sim_metrics': sim_metrics, 'grad_metrics': grad_metrics} 
        return data_frames 

    def write_transition_id(self, _uuid, manually_generated=False):
        'write a transition uuid to postgres'
        ## format bool as SQL 
        if manually_generated:
            manually_generated = 'TRUE'
        else:
            manually_generated = 'FALSE'
        ## generate cmd and exec 
        sql = f"INSERT INTO transitions VALUES ('{_uuid}', {manually_generated})" 
        self.__exec(sql) 
        pass
    
    def delete_non_manually_generated_transitions(self):
        sql = 'DELETE FROM transitions WHERE NOT manually_generated'
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

    def delete_latest_model(self): 
        'deletes SQL entry, not MinIO'
        sql = 'TRUNCATE latest_model;' 
        self.__exec(sql) 
        pass 
    pass
