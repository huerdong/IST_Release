import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle

class ServerMLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(nn.Module, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.fc1 = nn.Linear(self.input_size, self.hidden_size)
    self.bn1 = nn.BatchNorm1d(self.hidden_size, momentum=1.0, 
        affine=True, track_running_stats=False)
    self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    self.partition_index = [i for i in range(self.hidden_size)]

  def partition(self, device_num, device_hidden_layer_size):
    index_hidden_layer = []
    shuffle(self.partition_index)
    for i in range(device_num):
      index = torch.tensor(self.partition_index[i * device_hidden_layer_size:
          (i+1) * device_hidden_layer_size])
      index_hidden_layer.append(index)
    
    return index_hidden_layer
 
  def partition_single(self, device_hidden_layer_size):
    index_hidden_layer = []
    shuffle(self.partition_index)

    i = 0
    index = torch.tensor(self.partition_index[i * device_hidden_layer_size:
          (i+1) * device_hidden_layer_size])
    index_hidden_layer.append(index)
  
    return index_hidden_layer

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x)
    x = F.relu(x, inplace=True)
    x = self.fc2(x)
    x = F.log_softmax(x, dim=1)
    return x


