import torch
from torch import nn
import numpy as np
from collections import defaultdict
from copy import deepcopy
from utils.util import load_json,save_results,save_numpy_to_file,load_numpy_from_file

class HistoryMemory(nn.Module):

    def __init__(self, n_nodes, memory_dimension,raw_node_features, message_dimension=None,
                 device="cpu", combination_method='sum'):
        super(HistoryMemory, self).__init__()
        self.n_nodes = n_nodes
        self.raw_node_features = raw_node_features
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.combination_method = combination_method
        self.max_date_key = 2010
        self.__init_memory__()
   
    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        self.his_memory = nn.Parameter(self.raw_node_features.to(self.device), requires_grad=False)

    
    def get_history_memory(self, key, node_idxs):
        return self.history_memory[key][node_idxs, :]

    def backup_his_memory(self):
        return self.his_memory.data.clone()

    def restore_his_memory(self, memory_backup):
        self.his_memory.data = memory_backup.clone()
        
    def set_his_memory(self, key, values):
        self.his_memory[key] = values

    def detach_his_memory(self):
        self.his_memory.detach_()


 
