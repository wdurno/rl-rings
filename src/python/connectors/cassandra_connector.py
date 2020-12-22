from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel 
from cassandra.query import SimpleStatement 
import uuid
from strageABC import __StorageABC 

class CassandraConnector(__StorageABC):
    'Cassanda storage connector'
    
    def __init__(self, url): 
        'init storage interface instance'
        super().__init__(url, secret=None) 
        self.connection = None  
        pass 

    def __get_connection(self):
        'Get live connection to storage.'
        if self.connection is None:
            cluster = Cluster(['cassandra'], port=9042)  
            self.connection = cluster.connect()  
        return self.connection 

    def __exec(self, cql): 
        'submit a cql request to cassandra'
        connection = self.__get_connection() 
        try:
            return connection.execute(cql).all() 
        except: 
            ## assume connectivity error 
            self.connection = None 
            connection = self.__get_connection() 
            return connection.execute(cql).all()
        pass 

    def close_connection(self): 
        'Close any existing storage connection'
        raise NotImplementedError('Abstract base class not concretized!')

    def init_storage(self): 
        '''
        Initialize storage instance for the entire cluster. 
        You should only need to run this once. 
        '''
        cmd1 = '''
        CREATE  KEYSPACE IF NOT EXISTS cassandra  
           WITH REPLICATION = { 
              'class' : 'SimpleStrategy',
              'replication_factor' : 3  
           }
        '''
        cmd2 = '''
        CREATE TABLE IF NOT EXISTS cassandra.simulations (
            id uuid,
            PRIMARY KEY (id) 
        );
        ''' ## TODO finish data model 
        self.__exec(cmd1) 
        self.__exec(cmd2) 
        pass 
    
    ## other methods will be application and storage-specific  
