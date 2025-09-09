import h5py
import numpy as np

def add_gaussian_noise_to_array(arr, sigma, seed=None):

    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    else:
        noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    noisy = arr + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


file_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/filtered_chunk1_test.hdf5"
sigma = 0.08   
seed = 42      

with h5py.File(file_path, "r+") as f:  
    for video_id in f.keys():
        group = f[video_id]
        if "image" in group: 
            frames = group["image"][:]  

            frames_float = frames.astype(np.float32) / 255.0


            noisy_frames = add_gaussian_noise_to_array(frames_float, sigma, seed=seed)


            noisy_frames_u8 = (noisy_frames * 255.0).round().astype(np.uint8)


            del group["image"] 
            group.create_dataset("image", data=noisy_frames_u8, compression="gzip")