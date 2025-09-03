import h5py

#file_path_train = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/filtered_chunk1_train.hdf5"
#file_path_test = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/filtered_chunk1_test.hdf5"
file_path_val = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/filtered_chunk1_val.hdf5"
with h5py.File(file_path_val, "r") as f:
    print("Chiavi principali nel file HDF5 val:")
    all_keys = list(f.keys())
    print(all_keys)
    print(f"number of keys: {len(all_keys)}")
   