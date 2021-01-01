import base64
import pickle 

def unpack_obj(b64_str, in_bytes=False):
    'b64_str -> obj'
    if in_bytes:
        return pickle.loads(base64.b64decode(b64_str))
    return pickle.loads(base64.b64decode(b64_str.encode()))

def pack_obj(obj, out_bytes=False):
    'obj -> b64_str'
    if out_bytes:
        return base64.b64encode(pickle.dumps(obj)) 
    return base64.b64encode(pickle.dumps(obj)).decode()

