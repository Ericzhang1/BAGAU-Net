import numpy as np
import os
import SimpleITK as sitk
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from data_util import Utrecht_preprocessing, GE3T_preprocessing

rows_standard = 200
cols_standard = 200

class WMHChallengeDataset(Dataset):
    def __init__(self, directory, t1=False):
        directoryFlair = os.path.join(directory, "pre/FLAIR.nii.gz")
        directoryAtlas = "/input/t2_flair/result.nii"
        directoryT1 = os.path.join(directory, "pre/T1.nii.gz")
        #directoryMask = os.path.join(directory, "wmh.nii.gz")
        self.framework = []
        self.label = []
        flair_image = sitk.ReadImage(directoryFlair)
        flair_array = sitk.GetArrayFromImage(flair_image)

        if flair_array.shape[1] > rows_standard and flair_array.shape[2] > cols_standard: self.type = 0
        else: self.type = 1

        t1_image = sitk.ReadImage(directoryT1)
        t1_array = sitk.GetArrayFromImage(t1_image)
        atlas_image = sitk.ReadImage(directoryAtlas)
        atlas_array = sitk.GetArrayFromImage(atlas_image)
        #mask_image = sitk.ReadImage(directoryMask)
        #mask_array = sitk.GetArrayFromImage(mask_image)
        image_rows_Dataset, image_cols_Dataset = flair_array.shape[1], flair_array.shape[2]
        start_cut = 46
        if self.type == 1: 
            sample = GE3T_preprocessing(flair_array, t1_array)
            atlas = atlas_array.copy()
            atlas_array = np.ndarray((np.shape(atlas)[0], rows_standard, cols_standard), dtype=np.float32)
            atlas_array[...] = 0
            
            atlas_array[:, :, (cols_standard-image_cols_Dataset)//2:(cols_standard+image_cols_Dataset)//2] = atlas[:, start_cut:start_cut+rows_standard, :]

            #label = mask_array.copy()
            #mask_array = np.ndarray((np.shape(label)[0], rows_standard, cols_standard), dtype=np.float32)
            #mask_array[...] = 0
            #mask_array[:, :, (cols_standard-image_cols_Dataset)//2:(cols_standard+image_cols_Dataset)//2] = label[:, start_cut:start_cut+rows_standard, :]
        else: 
            sample = Utrecht_preprocessing(flair_array, t1_array)
            atlas_array = atlas_array[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
            #mask_array = mask_array[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
        if not t1:
            sample = sample[..., 0]
            sample = sample[..., np.newaxis]

        atlas_array = atlas_array[..., np.newaxis]
        #mask_array = mask_array[..., np.newaxis]
        sample = np.concatenate((sample, atlas_array), axis = -1)
        self.framework = sample
        self.label = np.zeros(self.framework.shape)

        #mask_image = sitk.ReadImage(directoryMask)
        #mask_array = sitk.GetArrayFromImage(mask_image)
        self.eval_mask = flair_array
        #self.eval_dir = directoryMask

        self.framework = np.transpose(self.framework, (0, 3, 1, 2))
        self.label = np.transpose(self.label, (0, 3, 1, 2))

    def __len__(self):
        return len(self.framework)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'image': self.framework[idx], 'mask': self.label[idx]}
