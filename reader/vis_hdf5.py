import h5py

hdf5_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/Similarity/probs_retinanet.hdf5"

with h5py.File(hdf5_path, "r") as f:
    print("=== Contenuto del file HDF5 ===")
    for key in f.keys():
        dset = f[key]
        print(f"Video ID: {key}")
        print(f"  Shape: {dset.shape}")   # (num_frames, num_concepts)
        print(f"  Dtype: {dset.dtype}")
        if dset.shape[0] > 0:
            print(f"  First frame probs (prime 5): {dset[0, :5]}")
        print("-" * 50)