from connectors import mc, cc, pc 

def get_latest_model(max_retries=2):
    '''
    Get latest model as produced by the paramter server. 
    Attempt at-most `max_retries` times. 
    If failed, return `None`.
    If succeeded, return a path (`str`) to the model file. 
    '''
    ## TODO implement parameter server 
    return None 

def upload_transition(transition):
    cc.insert_game_transition(transition) 
    pass

def upload_metrics():
    pass 
