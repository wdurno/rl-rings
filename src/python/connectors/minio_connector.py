
import io
from minio import Minio
from connectors.storageABC import __StorageABC 

class MinIOConnector(__StorageABC):
    'Connect to MinIO'
    
    def __init__(self, url, secret): 
        'init MinIO connector'
        ## check args 
        value_error = ValueError('secret must be a dict, {"accesskey": "...", "secretkey": "..."}') 
        if type(secret) != dict: 
            raise value_error
        if 'accesskey' not in secret or 'secretkey' not in secret: 
            raise value_error 
        ## store values 
        super().__init__(url, secret) 
        self.connection = None  
        pass 

    def __get_connection(self):
        'Get live connection to storage.'
        if self.connection is None: 
            self.connection = Minio(self.url, \
                    access_key=self.secret['accesskey'], \
                    secret_key=self.secret['secretkey'], \
                    secure=False) 
        return self.connection  
    
    def close_connection(self): 
        'Close any existing storage connection'
        ## I don't think connections actually persist
        ## So, enable a reconnect anyways 
        self.connection = None 
        pass

    def init_storage(self): 
        '''
        Initialize storage instance for the entire cluster. 
        You should only need to run this once. 
        '''
        connection = self.__get_connection() 
        connection.make_bucket('models') 
        print('MinIO bucket made: `models`') 
        pass

    def get(self, path, bucket='models'):
        'read a blob from storage'
        connection = self.__get_connection() 
        try:
            r = connection.get_object(bucket, path) 
            blob = r.read() ## TODO is this correct? 
        except Exception as e:
            raise e 
        finally: 
            r.close() 
            r.release_conn()
        return blob 

    def set(self, path, blob: bytes, bucket='models'): 
        'write a blob to storage'
        connection = self.__get_connection() 
        return connection.put_object(bucket, path, io.BytesIO(blob)) 
    pass 

