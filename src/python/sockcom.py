import socket 
from typing import Callable 

class Sockcom:
    '''
    simple socket communicator 
    message format: 
    [ message length ]LEN[ message of length `message length` ] 
    Example: 5LEN12345 
    '''
    def __init__(self, server_callback: Callable=None, host: str='0.0.0.0', port: int=8889, batch_size: int=1024, connection_timeout_seconds: float=0.1): 
        '''
        Initialize a Sockcom socket communicator instance. If serving, nitialization is forever blocking, always serving messages as recieved. 
        inputs:
         - server_callback: if serving, provide a callback function. It will be given a byte string of complete message. Please satisfy f(bytes) -> bytes. 
         - host: hostname to communicate with 
         - port: port number to communicate with 
         - batch_size: maximum size of all batches recieved. Must cover LEN of first message. 
         - connection_timeout_seconds: maximum wait time between byte packets before closing connection 
        '''
        ## store args 
        self.server_callback = server_callback 
        self.host = host 
        self.port = port 
        self.batch_size = batch_size 
        self.connection_timeout_seconds = connection_timeout_seconds 
        if server_callback is not None: 
            print(f'hosting sockcom server on {self.host}:{self.port}...') 
            serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            serversocket.bind((self.host, self.port)) 
            serversocket.listen(5) 
            while True: 
                (clientsocket, address) = serversocket.accept()
                try: 
                    ## connection recieved, get content 
                    message_bytes = self.__recv(clientsocket) 
                    return_bytes = self.server_callback(message_bytes) 
                    ## send response 
                    self.__send(clientsocket, return_bytes) 
                    ## conversation complete 
                    clientsocket.close() 
                except Exception as e: 
                    print('ERROR: '+str(e)) 
                pass 
            pass 
        pass 

    def send_to_server(self, message: bytes) -> bytes: 
        ## open connection 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        s.connect((self.host, self.port)) 
        self.__send(s, message) 
        ## get response 
        r = self.__recv(s) 
        s.close() 
        return r 

    def __send(self, clientsocket, message: bytes) -> None: 
        ## package message  
        l = len(message)    
        message = str(l).encode() + b'LEN' + message 
        ## send it          
        chunk_idx = 0       
        while chunk_idx * self.batch_size < len(message): 
            clientsocket.send(message[(chunk_idx*self.batch_size):((chunk_idx+1)*self.batch_size)]) 
            chunk_idx += 1  
            pass
        pass 

    def __recv(self, clientsocket) -> bytes: 
        ## get message length with first batch 
        clientsocket.settimeout(self.connection_timeout_seconds) 
        message_bytes = clientsocket.recv(self.batch_size) 
        len_idx = message_bytes.find(b'LEN') 
        total_message_length = int(message_bytes[:len_idx].decode()) + len_idx + 3 
        len_covered = len(message_bytes) 
        message_chunks = [message_bytes[len_idx + 3:]] # drop [ message length ]LEN 
        ## get rest of message 
        while len_covered < total_message_length: 
            clientsocket.settimeout(self.connection_timeout_seconds) 
            message_bytes = clientsocket.recv(self.batch_size) 
            len_covered += len(message_bytes) 
            message_chunks.append(message_bytes) 
            pass 
        return b''.join(message_chunks) 
    
    @staticmethod  
    def test_func(message: bytes) -> bytes:
        return str({message}).encode() 
    pass 

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser(description='for testing Sockcom') 
    parser.add_argument('--server', action='store_true', default=False, help='run as server, otherwise client') 
    args = parser.parse_args() 
    if args.server: 
        _ = Sockcom(server_callback=Sockcom.test_func, batch_size=10) 
    else: 
        r = Sockcom(batch_size=10).send_to_server(b'test-test-test') 
        print(r) 
