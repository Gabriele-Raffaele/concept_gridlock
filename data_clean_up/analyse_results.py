import pandas as pd
import matplotlib.pyplot as plt 
import torch
import numpy as np

def mse_loss(input, target, mask, reduction="mean"):
        out = (input[~mask]-target[~mask])**2
        return out.mean() if reduction == "mean" else out 

p_concept_large = '/kaggle/working/comma_angle_none_True_100.csv'
df = pd.read_csv(p_concept_large, header=None)
df.columns = ['preds', 'targets']

m = (df['targets'] > 40).astype(bool)  | (df['targets'] == 0).astype(bool) 
loss1 = torch.sqrt(mse_loss(torch.tensor(df['preds']),  torch.tensor(df['targets']), torch.tensor(m)))
m = (df['targets'] > 50).astype(bool)  | (df['targets'] == 0).astype(bool) 
loss2 = torch.sqrt(mse_loss(torch.tensor(df['preds']),  torch.tensor(df['targets']), torch.tensor(m)))
m = (df['targets'] > 60).astype(bool)  | (df['targets'] == 0).astype(bool) 
loss3 = torch.sqrt(mse_loss(torch.tensor(df['preds']),  torch.tensor(df['targets']), torch.tensor(m)))
print(loss1.item(), loss2.item(), loss3.item())