import pickle
import torch
import numpy 




pkl_path = pkl_path = "conc/b0c9d2329ad1606b_2018-07-29--11-17-20_3/00001.pkl"
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
print(data.keys())
print(f"Concepts {len(data['concepts'])}")
print(f"ego {data['ego'].shape}")
print(f"Main {data['main'].shape}")
logits_per_image = data['main'][:, 5:]