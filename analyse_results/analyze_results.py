import pandas as pd
import numpy
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import glob
import os
import matplotlib.pyplot as plt 
from PIL import Image
from collections import Counter
import warnings 
import glob
from vis_utils import get_regular_ckpt_from_lightning_checkpoint
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import pad_collate
from dataloader_comma import CommaDataset
from dataloader_nuscenes import NUScenesDataset
from model import VTN

gpu_num = 1
gpu = f'cuda:{gpu_num}'
multitask = 'distance'
backbone = 'none'
concept_features = True

commda_ds = CommaDataset(dataset_type="test",
        multitask=multitask,
        ground_truth="normal", dataset_path='/kaggle/input/filtered-chunk-hdf5')
dataloader_comma = DataLoader(commda_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=pad_collate)

model = VTN(multitask=multitask, backbone=backbone, concept_features=concept_features, device = f"cuda:{gpu_num}")

#Build checkpoint path -G.R.
ckpt_root = f"/kaggle/working/ckpts_final_comma_distance_none_True_1"
#find the latest version -G.R.
versions = glob.glob(os.path.join(ckpt_root, "lightning_logs", "version_*"))
if not versions:
    raise FileNotFoundError("None found")
latest_version = max(versions, key=os.path.getmtime)

# Find checkpoints in the latest version -G.R.
ckpt_files = glob.glob(os.path.join(latest_version, "checkpoints", "*.ckpt"))
if not ckpt_files:
    raise FileNotFoundError("None found.")
checkpoint_path = max(ckpt_files, key=os.path.getmtime)

print(f"Using checkpoint: {checkpoint_path}")

ckpt = torch.load(checkpoint_path)
state_dict = ckpt['state_dict']
state_dict = get_regular_ckpt_from_lightning_checkpoint(state_dict)
#added this to load the model correctly -G.R.
model.load_state_dict(state_dict)
print('done')

#what this module does it loads a pre-trained model and prepares it for evaluation on a test dataset -G.R.
