from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
import numpy as np
import torch
import h5py
from scipy import ndimage
import cv2

class CommaDataset(Dataset):
    def __init__(
        self,
        dataset_type="train",
        use_transform=False,
        multitask="angle",
        ground_truth="desired",
        return_full=False, 
        dataset_path =None,
        dataset_fraction=1.0
    ):
        assert dataset_type in ["train", "val", "test"]

        self.dataset_type = dataset_type
        self.dataset_fraction = dataset_fraction
        self.max_len = 240
        self.ground_truth = ground_truth
        self.multitask = multitask
        self.use_transform = use_transform
        self.return_full = return_full
        self.dataset_path = dataset_path
        #self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))
        self.resize = transforms.Resize((224,224))
        #self.resize = transforms.Resize((480,480))
        data_path = f"{self.dataset_path}/filtered_chunk1_{dataset_type}.hdf5" if ground_truth == "regular" else f"{self.dataset_path}/filtered_chunk1_{dataset_type}.hdf5"
        self.people_seqs = []
        self.h5_file = h5py.File(data_path, "r")
        self.keys = list(self.h5_file.keys())
            
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        person_seq = {}
        seq_key  = self.keys[idx]
        keys_ = self.h5_file[seq_key].keys()#'angle', 'brake', 'dist', 'gas', 'image', 'time', 'vEgo'
        file = self.h5_file
        
        for key in keys_:                        
            seq = file[seq_key][key][()]
            seq = seq if len(seq) <= 241 else seq[1::5]
            person_seq[key] = torch.from_numpy(np.array(seq[0:self.max_len]).astype(float)).type(torch.float32)
        sequences = person_seq
        distances = sequences['dist']
        distances = ndimage.median_filter(distances, size=128, mode='nearest')

        steady_state = ~np.array(sequences['gaspressed']).astype(bool) & ~np.array(sequences['brakepressed']).astype(bool) & ~np.array(sequences['leftBlinker']).astype(bool) & ~np.array(sequences['rightBlinker']).astype(bool)
        last_idx = 0
        desired_gap = np.zeros(distances.shape)

        for i in range(len(steady_state)-1):
            if steady_state[i] == True:
                desired_gap[last_idx:i] = int(distances[i])
                last_idx = i
        desired_gap[-12:] = distances[-12:].mean().item()

        distances = sequences['dist'] if self.ground_truth else desired_gap
        images = sequences['image']
        images = images[:,0:160, :,:]#crop the image to remove the view of the inside car console
        images = images.permute(0,3,1,2)
        if not self.return_full:
            images = self.normalize(images/255.0)
        else:
            images = images/255.0
        images = self.resize(images)
        images_cropped = images
        intervention = np.array(sequences['gaspressed']).astype(bool) | np.array(sequences['brakepressed']).astype(bool) 
        res = images_cropped, images_cropped,  sequences['vEgo'],  sequences['angle'], distances, seq_key
        if self.return_full: 
            return images_cropped,  sequences['vEgo'],  sequences['angle'], distances, np.array(sequences['gaspressed']).astype(bool),  np.array(sequences['brakepressed']).astype(bool) , np.array(sequences['CruiseStateenabled']).astype(bool), seq_key
        if self.multitask == "distance":
            res = images_cropped, images_cropped, sequences['vEgo'], sequences['angle'], distances, seq_key
        if self.multitask == "intervention":
            res = images_cropped, images_cropped, sequences['vEgo'], distances, torch.tensor(np.array(sequences['gaspressed']).astype(bool) | np.array(sequences['brakepressed']).astype(bool)), seq_key
        return res