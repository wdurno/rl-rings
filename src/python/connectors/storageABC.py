
from abc import ABC 

class __StorageABC(ABC):
    'Abstract storage class, standardizing interface'
    
    def __init__(self, url, secret): 
        'init storage interface instance' 
        self.url = url 
        self.secret = seret 
        pass 

    def __get_connection(self):
        'Get live connection to storage.'
        raise NotImplementedError('Abstract base class not concretized!') 

    def close_connection(self): 
        'Close any existing storage connection'
        raise NotImplementedError('Abstract base class not concretized!')

    def init_storage(self): 
        '''
        Initialize storage instance for the entire cluster. 
        You should only need to run this once. 
        '''
        raise NotImplementedError('Abstract base class not concretized!')
    
    ## other methods will be application and storage-specific  
