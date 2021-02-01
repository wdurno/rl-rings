from connectors import pc 
import pandas as pd

def get_avg_q_loss_by_model(): 
    metrics = pc.get_metrics() 
    sim_metrics = metrics['sim_metrics'] 
    grad_metrics = metrics['grad_metrics'] 
    model_idx = grad_metrics['model'].apply(__model_name_to_int) 
    grad_metrics['model_idx'] = model_idx 
    grouped = grad_metrics.groupby('model_idx').mean() 
    ## want model_idx as column, not index 
    df = pd.DataFrame({
        'model_idx': grouped.index, 
        'loss': grouped['loss'], 
        'q_pred': grouped['q_pred']}) 
    ## model_idx can exist in index too 
    df = df.reset_index(drop=True) 
    return df 

def get_avg_q_loss_reward(): 
    ## get reward and model idx
    metrics = pc.get_metrics() 
    sim_metrics = metrics['sim_metrics'] 
    sim_metrics['model_idx'] = sim_metrics['model'].apply(__model_name_to_int) 
    grouped = sim_metrics.groupby('model_idx').mean() 
    mdl_idx_reward = pd.DataFrame({
        'model_idx': grouped.index,
        'reward': grouped['reward']}) 
    ## model_idx is also index, dropping... 
    mdl_idx_reward = mdl_idx_reward.reset_index(drop=True) 
    ## get q, loss, model idx 
    mdl_idx_q_loss = get_avg_q_loss_by_model() 
    ## join 
    return pd.merge(mdl_idx_reward, mdl_idx_q_loss, how='inner', on='model_idx') 

def __model_name_to_int(name: str):
    'example: returns 12 from /models/model-12-DQN.pkl'
    return int(name.split('-')[1]) 

