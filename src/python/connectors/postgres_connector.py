
from storageABC import __StorageABC
import psycopg2

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
            self.connection = psycopg2.connect(dbname="api", user="postgres", host="db", port="5432", password=self.secret) 
        return self.connection 

    def __exec(self, sql): 
        'Execute sql. Connectivity abstracted-away.'
        print('POSTGRES EXEC: '+str(sql)) 
        ## define attempt mechanism 
        def run_sql(sql):
            connection = self.__get_connection() 
            cur = connection.cursor() 
            cur.execute(sql) 
            rows = cur.fetchall() 
            cur.commit() 
            return rows 
        try:
            ## first attempt, assuming connectivity 
            return run_sql(sql) 
        except psycopg2.OperationalError: 
            ## assuming connectivity error...
            ## attempting reconnection and trying again... 
            self.connection = None 
            retrun run_sql(sql) 
        pass 

    def close_connection(self): 
        'Close any existing storage connection'
        if self.connection is not None: 
            self.connection.close() 
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
        self.__exec(sql1)
        self.__exec(sql2)
        pass 
    pass 
