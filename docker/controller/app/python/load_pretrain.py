import minerl 

data = minerl.data.make('MineRLObtainDiamond-v0')
for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=32):
    print((current_state, action, reward, next_state, done)) 

