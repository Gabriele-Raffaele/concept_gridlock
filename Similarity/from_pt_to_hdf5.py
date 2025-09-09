import os
import torch
import h5py
import numpy as np

# --- CONFIG ---
input_folder = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/Similarity/concepts_with_noise"      # cartella con i file .pt
output_hdf5 = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/Similarity/probs_retinanet_noisy.hdf5"  # file HDF5 da creare


with h5py.File(output_hdf5, "w") as h5f:

    for fname in os.listdir(input_folder):
        if fname.endswith(".pt"):
            video_id = os.path.splitext(fname)[0]  # nome del video
            file_path = os.path.join(input_folder, fname)
            
        
            data = torch.load(file_path)
            

            concepts = data["concepts"][:, 1:]  # shape (240, 148)
            concepts_np = concepts.numpy()      # torch -> numpy
            
            h5f.create_dataset(video_id, data=concepts_np, compression="gzip")
            
            print(f"Salvato: {video_id} -> {concepts_np.shape}")

print("All .pt files have been saved to HDF5.")