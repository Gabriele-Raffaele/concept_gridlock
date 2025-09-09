import h5py
import numpy as np

def add_gaussian_noise_to_array(arr, sigma, seed=None):
    """
    arr: numpy array float32 in [0,1], shape (seq_len, H, W, C) o simile
    sigma: deviazione standard del rumore
    seed: opzionale per riproducibilità
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    else:
        noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    noisy = arr + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

# --- Parametri ---
file_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/filtered_chunk1_test.hdf5"
sigma = 0.08   # deviazione standard del rumore
seed = 42      # opzionale, per riproducibilità

with h5py.File(file_path, "r+") as f:  # "r+" per lettura/scrittura
    for video_id in f.keys():
        group = f[video_id]
        if "image" in group:  # usa la chiave corretta dei frame
            frames = group["image"][:]  # copia in memoria
            # Normalizziamo se necessario (dipende dal file)
            frames_float = frames.astype(np.float32) / 255.0

            # applichiamo rumore
            noisy_frames = add_gaussian_noise_to_array(frames_float, sigma, seed=seed)

            # rimappiamo a uint8
            noisy_frames_u8 = (noisy_frames * 255.0).round().astype(np.uint8)

            # sovrascriviamo il dataset
            del group["image"]  # eliminiamo quello vecchio
            group.create_dataset("image", data=noisy_frames_u8, compression="gzip")

print("Fatto: tutti i frame hanno ricevuto rumore Gaussian.")