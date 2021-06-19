import torch 
from ai.ai_runner import model, device, sample 
from ai.util import get_latest_model 

while True: 
    path_to_latest_model = get_latest_model() 
    model.load_state_dict(torch.load(path_to_latest_model)) 
    sample(model, device) 
