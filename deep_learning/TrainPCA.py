import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import os 
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
from data_loader import Diamond
from DeepRegress import RegressNet

num_epochs = 200
learning_rate = 1e-3
batch_size = 4
eval_interval = 5
save_interval = 100

trial_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir="tensorboard_logs/fit/" + trial_name
writer = SummaryWriter(log_dir)

train_csv = 'train_pca.csv'
val_csv = 'val_pca.csv'
parent_dir = os.path.abspath(os.path.join(os.getcwd(),os.pardir))
root_dir = os.path.join(parent_dir,'datasets')


train = Diamond(csv_file=train_csv,root_dir=root_dir)
trainload = DataLoader(train, batch_size=batch_size, shuffle=True)
val = Diamond(csv_file=val_csv,root_dir=root_dir)
valload = DataLoader(val, batch_size=batch_size//4, shuffle=True)

model = RegressNet(6,100,1)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay = 1e-5)
opt = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
model.train()
for epoch in range(num_epochs):
    epochs = epoch+1
    t_running_loss = 0.0
    for value in tqdm(trainload):
        optimizer.zero_grad()
        feature = value['feature']
        target = value['target']        
        out = model(feature)
        t_loss = loss_fn(out,target)
        t_loss.backward()
        optimizer.step()
        opt.step()
        
        t_loss_value = t_loss.detach().cpu()
        t_loss_scalar = t_loss_value.item()
        t_running_loss += t_loss_scalar
    
    t_record_loss = t_running_loss/len(trainload)
    writer.add_scalar('Train/Loss', t_record_loss, epoch)
    writer.flush()
    
    if epochs%eval_interval==0:
        model.eval()
        with torch.no_grad():
            v_running_loss = 0.0
            for test in tqdm(valload):
                feature = test['feature']
                target = test['target']
                out = model(feature)
                v_loss = loss_fn(out,target)
                v_loss_value = v_loss.detach().cpu()
                v_loss_scalar = v_loss_value.item()
                v_running_loss += v_loss_scalar
        
        v_record_loss = v_running_loss/len(valload)        
        writer.add_scalar('Val/Loss', v_record_loss,epoch)
        writer.flush()
    
    print('epoch: {}, loss: {}'.format(epochs, t_record_loss))  
    
    if epochs%save_interval == 0:
        Save_Name = trial_name + '_{}.pth'.format(epochs)
        Save_Rootdir = os.path.join(os.getcwd(), 'training_checkpoints')
        Save_Path = os.path.join(Save_Rootdir, Save_Name)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(), 'epoch':epochs}, Save_Path)