
from connectors.storageABC import __StorageABC
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT 

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
        sql4 = 'CREATE EXTENSION IF NOT EXISTS tsm_system_rows;'
        self.connection = psycopg2.connect(user='postgres', host=self.url, port='5432', password=self.secret)
        self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) 
        self.__exec(sql1, debug=True)
        self.close_connection() 
        self.__exec(sql2, debug=True)
        self.__exec(sql3, debug=True) 
        self.__exec(sql4, debug=True) 
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
        prob = min(100, 100*expected_transitions/total_transitions) 
        sql = f'SELECT transition_id FROM transitions TABLESAMPLE BERNOULLI({prob});'
        uuids = [r[0] for r in self.__exec(sql)] 
        return uuids 
    pass 
