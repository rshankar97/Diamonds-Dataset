import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os 
from tqdm import tqdm

class Diamond(Dataset):
    def __init__(self,csv_file, root_dir):
        self.csv_file = os.path.join(root_dir, csv_file)
        self.input = pd.read_csv(self.csv_file)
        self.index_x = [str(ind) for ind in self.input.keys() if str(ind)!='price']
        self.y_input = self.input['price']
        self.x_input = self.input[self.index_x]
    def __len__(self):
        return len(self.input)
    def __getitem__(self,idx):
        y_val = self.y_input.loc[idx]
        X_val = self.x_input.loc[idx]
        
        X_vec = np.array(X_val, dtype = np.float32)
        feature = torch.from_numpy(X_vec)
        
        y_arr = np.array(y_val, dtype = np.float32)
        target = torch.from_numpy(y_arr)
        target = torch.unsqueeze(target,dim=0)
        
        return {'feature':feature,'target':target}