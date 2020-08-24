import numpy as np
import os
import SimpleITK as sitk
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from data_util import Utrecht_preprocessing, GE3T_preprocessing, augmentation, augmentation2, augmentation3, augmentation4

rows_standard = 200
cols_standard = 200

class WMHChallengeDataset(Dataset):
    """Whitem atter hyperintensity challenge dataset"""
    def __init__(self, directory, train, test_subject, aug=True, domain_knowledge=True, aging=True, T1=True):
        self.dir = directory
        ge3t_path = os.path.join(self.dir, "GE3T")
        singapore_path = os.path.join(self.dir, "Singapore")
        utrecht_path = os.path.join(self.dir, "Utrecht")
        self.ge3t = os.listdir(ge3t_path)
        self.singapore = os.listdir(singapore_path)
        self.utrecht = os.listdir(utrecht_path)
        #Get rid of '.DS_Store' file
        self.ge3t = [x for x in self.ge3t if '.' not in x]
        self.singapore = [x for x in self.singapore if '.' not in x]
        self.utrecht = [x for x in self.utrecht if '.' not in x]
        self.ge3t.sort()
        self.singapore.sort()
        self.utrecht.sort()
        patient = len(self.ge3t) + len(self.singapore) + len(self.utrecht)
        self.framework = []
        self.label = []
        self.test_subject = test_subject
        for idx in range(patient):
            if idx < 20: 
                directoryFlair = os.path.join(self.dir, "GE3T/{}/pre/FLAIR.nii.gz".format(self.ge3t[idx%20]))
                directoryT1 = os.path.join(self.dir, "GE3T/{}/pre/T1.nii.gz".format(self.ge3t[idx%20]))
                directoryMask = os.path.join(self.dir, "GE3T/{}/wmh.nii.gz".format(self.ge3t[idx%20]))
                directoryAtlas = os.path.join(self.dir, "GE3T/{}/pre/result.nii".format(self.ge3t[idx%20]))
                directoryAtlas2 = os.path.join(self.dir, "GE3T/{}/pre/result2.nii".format(self.ge3t[idx%20]))
            elif idx < 40: 
                directoryFlair = os.path.join(self.dir, "Singapore/{}/pre/FLAIR.nii.gz".format(self.singapore[idx%20]))
                directoryT1 = os.path.join(self.dir, "Singapore/{}/pre/T1.nii.gz".format(self.singapore[idx%20]))
                directoryMask = os.path.join(self.dir, "Singapore/{}/wmh.nii.gz".format(self.singapore[idx%20]))
                directoryAtlas = os.path.join(self.dir, "Singapore/{}/pre/result.nii".format(self.singapore[idx%20]))
                directoryAtlas2 = os.path.join(self.dir, "Singapore/{}/pre/result2.nii".format(self.singapore[idx%20]))
            else: 
                directoryFlair = os.path.join(self.dir, "Utrecht/{}/pre/FLAIR.nii.gz".format(self.utrecht[idx%20]))
                directoryT1 = os.path.join(self.dir, "Utrecht/{}/pre/T1.nii.gz".format(self.utrecht[idx%20]))
                directoryMask = os.path.join(self.dir, "Utrecht/{}/wmh.nii.gz".format(self.utrecht[idx%20]))
                directoryAtlas = os.path.join(self.dir, "Utrecht/{}/pre/result.nii".format(self.utrecht[idx%20]))
                directoryAtlas2 = os.path.join(self.dir, "Utrecht/{}/pre/result2.nii".format(self.utrecht[idx%20]))

            #Read the data using sitk and convert into array
            flair_image = sitk.ReadImage(directoryFlair)
            t1_image = sitk.ReadImage(directoryT1)
            flair_array = sitk.GetArrayFromImage(flair_image)
            t1_array = sitk.GetArrayFromImage(t1_image)
            mask_image = sitk.ReadImage(directoryMask)
            mask_array = sitk.GetArrayFromImage(mask_image)
            if domain_knowledge:
                atlas_image = sitk.ReadImage(directoryAtlas)
                atlas_array = sitk.GetArrayFromImage(atlas_image)
            if aging:
                atlas_image = sitk.ReadImage(directoryAtlas2)
                atlas_array2 = sitk.GetArrayFromImage(atlas_image)
            
            if idx < 20: sample = GE3T_preprocessing(flair_array, t1_array)
            else: sample = Utrecht_preprocessing(flair_array, t1_array)
            #Take only Flair if T1 is not specified
            if not T1:
                sample = sample[..., 0]
                sample = sample[..., np.newaxis]

            image_rows_Dataset, image_cols_Dataset = mask_array.shape[1], mask_array.shape[2]
            start_cut = 46
            if idx < 20:
                label = mask_array.copy()
                mask_array = np.ndarray((np.shape(label)[0], rows_standard, cols_standard), dtype=np.float32)
                mask_array[...] = 0
                mask_array[:, :, (cols_standard-image_cols_Dataset)//2:(cols_standard+image_cols_Dataset)//2] = label[:, start_cut:start_cut+rows_standard, :]
                if domain_knowledge:
                    #preprocess for atlas
                    atlas = atlas_array.copy()
                    atlas_array = np.ndarray((np.shape(atlas)[0], rows_standard, cols_standard), dtype=np.float32)
                    atlas_array[...] = 0
                    atlas_array[:, :, (cols_standard-image_cols_Dataset)//2:(cols_standard+image_cols_Dataset)//2] = atlas[:, start_cut:start_cut+rows_standard, :]
                if aging:
                    #more atlas processing
                    atlas2 = atlas_array2.copy()
                    atlas_array2 = np.ndarray((np.shape(atlas2)[0], rows_standard, cols_standard), dtype=np.float32)
                    atlas_array2[...] = 0
                    atlas_array2[:, :, (cols_standard-image_cols_Dataset)//2:(cols_standard+image_cols_Dataset)//2] = atlas2[:, start_cut:start_cut+rows_standard, :]
            else:
                mask_array = mask_array[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
                #preprocess for atlas
                if domain_knowledge:
                    atlas_array = atlas_array[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
                if aging:
                    atlas_array2 = atlas_array2[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]

            mask_array = mask_array[..., np.newaxis]
            if domain_knowledge:
                atlas_array = atlas_array[..., np.newaxis]
                sample = np.concatenate((sample, atlas_array), axis = -1)
            
            if aging:
                atlas_array2 = atlas_array2[..., np.newaxis]
                sample = np.concatenate((sample, atlas_array2), axis = -1)

            #set the test subject 
            if not train and idx == test_subject:
                self.eval_sample = sample
            
            if len(self.framework) == 0:
                self.framework = sample[10:-10]
                self.label = mask_array[10:-10]
                continue
            if idx < 20:
                self.framework = np.concatenate((self.framework, sample[10:-10]), axis=0)
                self.label = np.concatenate((self.label, mask_array[10:-10]), axis=0)
            else:
                self.framework = np.concatenate((self.framework, sample[5:-5]), axis=0)
                self.label = np.concatenate((self.label, mask_array[5:-5]), axis=0)

        #save the processed results
        #np.save('image.npy', self.framework)
        #np.save('label.npy', self.label)

        if train:
            if isinstance(test_subject, int):
                if test_subject < 20:
                    self.framework = np.delete(self.framework, range(test_subject*63, (test_subject+1)*63), axis=0)
                    self.label = np.delete(self.label, range(test_subject*63, (test_subject+1)*63), axis=0)
                else:
                    self.framework = np.delete(self.framework, range(1260+(test_subject-20)*38, 1260+(test_subject-19)*38), axis=0)
                    self.label = np.delete(self.label, range(1260+(test_subject-20)*38, 1260+(test_subject-19)*38), axis=0)
            elif isinstance(test_subject, list):
                test_subject = [int(x) for x in test_subject]
                trainset = []
                for i in range(60):
                    if i in test_subject:
                        continue
                    if i < 20:
                        image = self.framework[i*63:(i+1)*63, ...]
                        label = self.label[i*63:(i+1)*63, ...]
                    else:
                        image = self.framework[1260+(i-20)*38:1260+(i-19)*38, ...]
                        label = self.label[1260+(i-20)*38:1260+(i-19)*38, ...]
                    if len(trainset) == 0:
                        trainset = image
                        trainset_label = label
                        continue
                    trainset = np.concatenate((trainset, image), axis=0)
                    trainset_label = np.concatenate((trainset_label, label), axis=0)
                self.framework = np.asarray(trainset)
                self.label = np.asarray(trainset_label)
            else:
                print(f"test subject type: {test_subject} unknown: int or list accepted.")
                sys.exit(1)
            if aug:
                #mirror
                self.framework = np.concatenate((self.framework, self.framework[..., ::-1, :]), axis=0)
                self.label = np.concatenate((self.label, self.label[..., ::-1, :]), axis=0)
                #scale, shear, rotate
                images_aug = np.zeros(self.framework.shape, dtype=np.float32)
                masks_aug = np.zeros(self.label.shape, dtype=np.float32)
                for i in range(self.framework.shape[0]):
                    variants = T1 + domain_knowledge + aging + 1
                    if variants == 1:
                        images_aug[i, ..., 0], masks_aug[i, ..., 0] = \
                                augmentation4(self.framework[i, ..., 0], self.label[i, ..., 0])
                    elif variants == 2:
                        images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = \
                                augmentation(self.framework[i, ..., 0], self.framework[i, ..., 1], self.label[i, ..., 0])
                    elif variants == 3:
                        images_aug[i, ..., 0], images_aug[i, ..., 1], images_aug[i, ..., 2], masks_aug[i, ..., 0] = \
                                augmentation2(self.framework[i, ..., 0], self.framework[i, ..., 1], self.framework[i, ..., 2], self.label[i, ..., 0])
                    else:
                        images_aug[i, ..., 0], images_aug[i, ..., 1], images_aug[i, ..., 2], images_aug[i, ..., 3], masks_aug[i, ..., 0] = \
                                augmentation3(self.framework[i, ..., 0], self.framework[i, ..., 1], self.framework[i, ..., 2], self.framework[i, ..., 3], self.label[i, ..., 0])

                self.framework = np.concatenate((self.framework, images_aug), axis=0)
                self.label = np.concatenate((self.label, masks_aug), axis=0)
        else:
            self.framework = self.eval_sample
            self.label = np.zeros(self.framework.shape)
            #Set up the directory for evaluation
            if test_subject < 20:
                self.eval_dir = self.dir + "/GE3T/{}/wmh.nii.gz".format(self.ge3t[test_subject%20])
            elif test_subject < 40:
                self.eval_dir = self.dir + "/Singapore/{}/wmh.nii.gz".format(self.singapore[test_subject%20])
            else:
                self.eval_dir = self.dir + "/Utrecht/{}/wmh.nii.gz".format(self.utrecht[test_subject%20])
            eval_mask = sitk.ReadImage(self.eval_dir)
            self.eval_mask = sitk.GetArrayFromImage(eval_mask)
                
        self.framework = np.transpose(self.framework, (0, 3, 1, 2))
        self.label = np.transpose(self.label, (0, 3, 1, 2))

    def __len__(self):
        return len(self.framework)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'image': self.framework[idx], 'mask': self.label[idx]}

if __name__ == "__main__":
    wmh_dataset = WMHChallengeDataset(directory="../raw", train=True, test_subject=[0,1,2,3], aug=True, domain_knowledge=True, aging=False, T1=True)
    dataloader = DataLoader(wmh_dataset, batch_size=4, shuffle=True, num_workers=4)
    assert len(wmh_dataset.framework) == len(wmh_dataset.label)
    print(wmh_dataset.framework.shape)
    print(wmh_dataset.label.shape)
   
    
            
