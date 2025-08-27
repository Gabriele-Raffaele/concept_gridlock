import h5py

file_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/concept_gridlock/filtered_chunk1_train.hdf5"

with h5py.File(file_path, "r") as f:
    print("Chiavi principali nel file HDF5:")
    all_keys = list(f.keys())

    with open("kept_seqs_modified.txt", "r") as txt_file:
        kept_seqs = [line.strip() for line in txt_file.readlines()]

    keys_in_txt = [key for key in all_keys if key in kept_seqs]
    keys_not_in_txt = [key for key in all_keys if key not in kept_seqs]

    print(f"Chiavi presenti sia nel file HDF5 che in kept_seqs.txt: {len(keys_in_txt)}")
    print(f"Chiavi presenti nel file HDF5 ma mancanti in kept_seqs.txt: {len(keys_not_in_txt)}")
    print("Chiavi mancanti in kept_seqs.txt:")
    for key in keys_not_in_txt:
        print(f" - {key}")
    '''
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
    print(f"\nShape: {data.shape}, dtype: {data.dtype}")'''
