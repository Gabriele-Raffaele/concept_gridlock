import h5py
import numpy as np
import matplotlib.pyplot as plt
file_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/filtered_chunk1_train.hdf5"
#file_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/filtered_chunk1_test.hdf5"
#file_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/filtered_chunk1_val.hdf5"
  
zero_counts = []
video_ids = []

with h5py.File(file_path, "r") as f:
    for video_id in f.keys():
        group = f[video_id]
        if "dist" in group:   # usa "distance" se è la chiave corretta
            distances = group["dist"][:].flatten()
            zero_count = np.sum(distances == 0.0)
            zero_counts.append(zero_count)
            video_ids.append(video_id)

# statistiche
zero_counts = np.array(zero_counts)
avg_zeros = zero_counts.mean()
min_zeros = zero_counts.min()
max_zeros = zero_counts.max()
video_max = video_ids[zero_counts.argmax()]

print(f"Numero medio di valori == 0.0 per video: {avg_zeros:.2f}")
print(f"Minimo: {min_zeros}, Massimo: {max_zeros} (video {video_max})")

# grafico
plt.figure(figsize=(12, 6))
plt.bar(range(len(video_ids)), zero_counts)
plt.xlabel("Video index")
plt.ylabel("Count di distance == 0.0")
plt.title("Numero di valori 0.0 per video")
plt.show()