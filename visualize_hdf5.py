import h5py

file_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/concept_gridlock/filtered_chunk1_test.hdf5"

with h5py.File(file_path, "r") as f:
    print("Chiavi principali nel file HDF5:")
    for key in f.keys():
        print(f" - {key}")
    
    # Accedi al gruppo
    group = f['val_video']
    print("\nContenuto del gruppo 'val_video':")
    for subkey in group.keys():
        print(f" - {subkey}")

    # Se vuoi leggere un dataset specifico dentro il gruppo
    first_dataset_name = list(group.keys())[-8]
    data = group[first_dataset_name][:]
    
    print(f"\nContenuto del dataset '{first_dataset_name}':")
    print(data)
    print(f"\nShape: {data.shape}, dtype: {data.dtype}")