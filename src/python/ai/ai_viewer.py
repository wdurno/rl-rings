import torch 
from ai.ai_runner import model, device, sample 
from ai.util import get_latest_model 

while True: 
    try:
        path_to_latest_model = get_latest_model() 
        model.load_state_dict(torch.load(path_to_latest_model)) 
    except Exception as e:
        print('Could not load model, using local model.\n'+str(e)) 
        pass 
    sample(model, device) 
