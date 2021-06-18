from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel 
from cassandra.query import SimpleStatement 
from connectors.storageABC import __StorageABC 
from connectors.util import pack_obj, unpack_obj 
import uuid
import pickle 
import base64

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

    def __exec(self, cql, debug=False, async_req=False): 
        'submit a cql request to cassandra'
        if debug: 
            if len(cql) < 500:
                print('CASSANDRA EXEC: '+str(cql)) 
            else:
                print('CASSANDRA EXEC: '+str(cql[:100])+'... (truncated), total size: '+str(len(cql))) 
        connection = self.__get_connection() 
        try:
            if async_req: 
                return connection.execute_async(cql) 
            else: 
                return connection.execute(cql).all() 
        except: 
            ## assume connectivity error 
            self.connection = None 
            connection = self.__get_connection() 
            if async_req:
                return connection.execute_async(cql) 
            else: 
                return connection.execute(cql).all()
        pass 

    def close_connection(self): 
        'Close any existing storage connection'
        ## TODO verify this mechanism 
        self.connection = None 
        pass

    def init_storage(self): 
        '''
        Initialize storage instance for the entire cluster. 
        You should only need to run this once. 
        '''
        cmd1 = '''
        CREATE  KEYSPACE IF NOT EXISTS cassandra  
           WITH REPLICATION = { 
              'class' : 'SimpleStrategy',
              'replication_factor' : 2  
           }
        '''
        cmd2 = '''
        CREATE TABLE IF NOT EXISTS cassandra.simulations (
            id uuid PRIMARY KEY,
            b64data text  
        );
        '''  
        self.__exec(cmd1, debug=True) 
        self.__exec(cmd2, debug=True) 
        pass 
    
    def insert_game_transition(self, obj): 
        'upload a single game transition'
        ## get base64 representation 
        obj_b64_string = pack_obj(obj)  
        ## generate key 
        _uuid = str(uuid.uuid1()) 
        ## upload 
        cmd = f"INSERT INTO cassandra.simulations (id, b64data) VALUES ({_uuid}, '{obj_b64_string}');"
        self.__exec(cmd) 
        return _uuid 

    def get_game_transition(self, _uuid):
        'get a single game transition'
        ## download data 
        cmd = f"SELECT b64data FROM cassandra.simulations WHERE id={_uuid};"
        row_list = self.__exec(cmd) 
        ## unpack single item 
        obj_b64_string = row_list[0].b64data 
        return unpack_obj(obj_b64_string) 

    def get_all_game_transition_uuids(self):
        ## download data 
        cmd = "SELECT id FROM cassandra.simulations;"
        row_list = self.__exec(cmd) 
        ## unpack 
        return [row.id for row in row_list]

    def get_transitions(self, uuid_list):
        return self.__get_objs(uuid_list, 'simulations') 

    def __get_objs(self, uuid_list, table, return_b64str=False):
        'returns results in order'
        ## start async requests 
        async_responses = [] 
        for _uuid in uuid_list: 
            cql = f'SELECT b64data FROM cassandra.{table} WHERE id={_uuid};' 
            async_response = self.__exec(cql) 
            async_responses.append(async_response) 
        ## get async responses 
        objs = [] 
        for response in async_responses:
            if type(response) != list: 
                ## wait for response 
                b64_str_list = response.result() 
            else:
                b64_str_list = response 
            ## skip empty responses 
            if len(b64_str_list) > 0: 
                ## first row, first column 
                b64_str = b64_str_list[0][0] 
                obj = b64_str
                if not return_b64str: 
                    obj = unpack_obj(b64_str) 
                    pass 
                objs.append(obj) 
                pass 
        return objs
    pass 





