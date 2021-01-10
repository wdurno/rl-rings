import io
from minio import Minio
from connectors.storageABC import __StorageABC
from connectors.util import pack_obj, unpack_obj

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
        connection.make_bucket('gradients') 
        print('MinIO bucket made: `gradients`') 
        pass

    def get(self, path, bucket='models'):
        'read a blob from storage'
        connection = self.__get_connection()
        r = None 
        try:
            r = connection.get_object(bucket, path) 
            blob = r.read() 
        except Exception as e:
            raise e 
        finally: 
            if r is not None: 
                r.close() 
                r.release_conn()
        return blob 

    def set(self, path, blob: bytes, bucket='models'): 
        'write a blob to storage'
        connection = self.__get_connection() 
        connection.put_object(bucket, path, io.BytesIO(blob), len(blob))
        pass 

    def set_gradient(self, _uuid, grad):
        'DEPRECATED!'
        self.set(str(_uuid), pack_obj(grad, out_bytes=True), bucket='gradients')
        pass

    def get_gradient(self, _uuid):
        'DEPRECATED!'
        return unpack_obj(self.get(str(_uuid), bucket='gradients'), in_bytes=True) 
    pass 

