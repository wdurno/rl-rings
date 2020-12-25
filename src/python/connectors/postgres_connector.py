
from storageABC import __StorageABC
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

    def __exec(self, sql): 
        'Execute sql. Connectivity abstracted-away.'
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
        self.connection = psycopg2.connect(user='postgres', host=self.url, port='5432', password=self.secret)
        self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) 
        self.__exec(sql1)
        self.close_connection() 
        self.__exec(sql2)
        pass 
    pass 
