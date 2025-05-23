from utils import * 
from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
import numpy as np
import torch
import h5py
from scipy import ndimage
import cv2

class NUScenesDataset(Dataset):
    def __init__(
        self,
        dataset_type="train",
        use_transform=False,
        multitask="angle",
        ground_truth="desired",
        return_full=False,
        max_len=240,
        dataset_path=None,
        dataset_fraction=1.0,
    ):
        assert dataset_type in ["train", "val", "test"]
        self.dataset_type = dataset_type
        self.ground_truth = ground_truth
        self.multitask = multitask
        self.max_len = max_len
        self.use_transform = use_transform
        self.dataset_fraction = dataset_fraction
        self.return_full = return_full
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.resize = transforms.Resize((224,224))
        data_path = f'{dataset_path}/nuscenes/test_mini_nuscenes.hdf5' if dataset_type == "val" else (f'{dataset_path}/nuscenes/train_consec_nuscenes.hdf5' if dataset_type == "train" else f'{dataset_path}/nuscenes/train_2_2_consec_nuscenes.hdf5')
        self.h5_file = h5py.File(data_path, "r")
        self.keys = list(self.h5_file.keys())
        corrupt_idx = 24
        if dataset_type == "train":
            self.keys.pop(corrupt_idx)
        
        '''if dataset_type == "test":
            p = f'{dataset_path}/nuscenes/test_2_consec_nuscenes.hdf5'
            self.h5_file2 = h5py.File(p, "r")
            self.keys2 = list(self.h5_file2.keys())
            p = f'{dataset_path}/nuscenes/test_3_consec_nuscenes.hdf5'
            self.h5_file3 = h5py.File(p, "r")
            self.keys3 = list(self.h5_file3.keys())'''

        
           
    def __len__(self):
        return int(len(self.keys) * self.dataset_fraction) if self.dataset_type == "train" else len(self.keys) #+ len(self.keys2) + len(self.keys3) 

    def __getitem__(self, idx):
        person_seq = {}
        if idx < len(self.keys):
            seq_key  = self.keys[idx]
            keys_ = self.h5_file[seq_key].keys()#'steering', 'brake', 'available_distance', 'image', 'utime', 'vehicle_speed'
            file = self.h5_file
        '''elif i < (len(self.keys) + len(self.keys2)):
            idx = idx - len(self.keys)
            seq_key  = self.keys2[idx]
            keys_ = self.h5_file2[seq_key].keys()#'steering', 'brake', 'available_distance', 'image', 'utime', 'vehicle_speed'
            file = self.h5_file2
        else:
            idx = idx - len(self.keys) - len(self.keys2)
            seq_key  = self.keys3[idx]
            keys_ = self.h5_file3[seq_key].keys()#'steering', 'brake', 'available_distance', 'image', 'utime', 'vehicle_speed'
            file = self.h5_file3'''
        
        for key in keys_:   
            if key == 'description': continue                    
            seq = file[seq_key][key][()]
            person_seq[key] = torch.from_numpy(np.array(seq[0:self.max_len]).astype(float)).type(torch.float32)
        sequences = person_seq
        distances = sequences['available_distance']

        steady_state =  ~np.array(sequences['brake']).astype(bool) & ~np.array(sequences['left_signal']).astype(bool) & ~np.array(sequences['right_signal']).astype(bool)
        last_idx = 0
        desired_gap = np.zeros(distances.shape)

        for i in range(len(steady_state)-1):
            if steady_state[i] == True:
                desired_gap[last_idx:i] = int(distances[i])
                last_idx = i
        desired_gap[-12:] = distances[-12:].mean().item()

        distances = sequences['available_distance'] if self.ground_truth == "normal" else desired_gap
        images = sequences['image']
        images = images.permute(0,3,1,2)
        if not self.return_full:
            images = self.normalize(images/255.0)
        else:
            images = images/255.0
        images = self.resize(images)
        images_cropped = images
        res = images_cropped, images_cropped,  sequences['vehicle_speed'],  sequences['steering'], distances
        intervent = ~steady_state
        if self.return_full: 
            return images_cropped,  sequences['vehicle_speed'],  sequences['steering'], distances, None,  np.array(sequences['brake']).astype(bool) , intervent
        if self.multitask == "distance":
            res = images_cropped, images_cropped, sequences['vehicle_speed'], distances, sequences['steering']
        return res 