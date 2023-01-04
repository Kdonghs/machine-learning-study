import torch
import torch.nn as nn

input_size = 5
hidden_size = 8
batch = 1
time_step = 10

####################################RNN

inputs = torch.Tensor(batch,time_step,input_size)
cell = nn.RNN(input_size,hidden_size,batch_first=True)
outputs,_status = cell(inputs)

print(outputs.shape)
print(_status.shape)

####################################DRNN

