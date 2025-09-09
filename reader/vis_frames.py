import h5py
import numpy as np

# --- CONFIG ---
hdf5_clean = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/Similarity/probs_retinanet.hdf5"  # file originale clean
hdf5_noisy = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/Similarity/probs_retinanet_noisy.hdf5"  # file originale noisy

video_id = "b0c9d2329ad1606b_2018-08-17--14-55-39_4"

def is_problematic_frame(c, n):
    return (
        np.isnan(c).any() or np.isnan(n).any() or
        np.isinf(c).any() or np.isinf(n).any() or
        np.var(c) == 0 or np.var(n) == 0
    )

with h5py.File(hdf5_clean, "r") as f_clean, h5py.File(hdf5_noisy, "r") as f_noisy:
    clean = f_clean[video_id][...]  # shape (frames, concepts)
    noisy = f_noisy[video_id][...]
    
    print(f"Video {video_id}: shape clean {clean.shape}, shape noisy {noisy.shape}")
    
    # Loop sui frame
    for i, (c, n) in enumerate(zip(clean, noisy)):
        if is_problematic_frame(c, n):
            print(f"\n[DEBUG] Frame {i} is problematic")
            print("Clean logits:", c)
            print("Noisy logits:", n)