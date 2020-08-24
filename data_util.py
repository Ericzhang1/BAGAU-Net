import numpy as np
import scipy.ndimage
import os
import sys
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
smooth=1.

def Utrecht_preprocessing(FLAIR_image, T1_image):

    channel_num = 2
    #print(np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
    FLAIR_image = FLAIR_image[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
    brain_mask_FLAIR = brain_mask_FLAIR[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
    ###------Gaussion Normalization here
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
    T1_image = T1_image[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
    brain_mask_T1 = brain_mask_T1[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
    #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])
    #---------------------------------------------------
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    T1_image  = T1_image[..., np.newaxis]
    imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis = -1)
    #print(np.shape(imgs_two_channels))
    return imgs_two_channels

def GE3T_preprocessing(FLAIR_image, T1_image):

    channel_num = 2
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
  
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])

    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    FLAIR_image_suitable[:, :, (cols_standard//2-image_cols_Dataset//2):(cols_standard//2+image_cols_Dataset//2)] = FLAIR_image[:, start_cut:start_cut+rows_standard, :]
   
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
 
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])

    T1_image_suitable[...] = np.min(T1_image)
    T1_image_suitable[:, :, (cols_standard-image_cols_Dataset)//2:(cols_standard+image_cols_Dataset)//2] = T1_image[:, start_cut:start_cut+rows_standard, :]
    #---------------------------------------------------
    FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable  = T1_image_suitable[..., np.newaxis]
    
    imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = -1)
    #print(np.shape(imgs_two_channels))
    return imgs_two_channels

def augmentation(x_0, x_1, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], x_1[..., 0], y[..., 0]

def augmentation2(x_0, x_1, x_2, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    x_2 = apply_transform(x_2[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], x_1[..., 0], x_2[..., 0], y[..., 0]

def augmentation3(x_0, x_1, x_2, x_3, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    x_2 = apply_transform(x_2[..., np.newaxis], transform_matrix, channel_axis=2)
    x_3 = apply_transform(x_3[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], x_1[..., 0], x_2[..., 0], x_3[..., 0], y[..., 0]

def augmentation4(x_0, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], y[..., 0]

def Utrecht_postprocessing(FLAIR_array, pred):
    start_slice = 6
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:,(image_rows_Dataset-rows_standard)//2:(image_rows_Dataset+rows_standard)//2,(image_cols_Dataset-cols_standard)//2:(image_cols_Dataset+cols_standard)//2] = pred[:,:,:,0]
    
    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred

def GE3T_postprocessing(FLAIR_array, pred):
    start_slice = 11
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:, start_cut:start_cut+rows_standard,:] = pred[:,:, (rows_standard-image_cols_Dataset)//2:(rows_standard+image_cols_Dataset)//2,0]

    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred

def split(validate_subjects=[], test_subjects = [], domain_knowledge=True, aging=True):
    images = np.load('image.npy')
    labels = np.load('label.npy')
    if not domain_knowledge and not aging:
        images = images[..., 0:2]
        labels = labels[..., 0:2]
    elif domain_knowledge and not aging:
        images = np.concatenate((images[..., 0:2], images[..., 2]), axis=0)
        labels = np.concatenate((labels[..., 0:2], labels[..., 2]), axis=0)
    elif not domain_knowledge and aging:
        images = np.concatenate((images[..., 0:2], images[..., 3]), axis=0)
        labels = np.concatenate((labels[..., 0:2], labels[..., 3]), axis=0)
    train, validate, test = [], [] ,[]
    train_label, validate_label, test_label = [], [] ,[]
    for i in range(60):
        unit = 63 if i < 20 else 38
        base = 0 if i < 20 else 1260
        slide = i if i < 20 else i - 20
        image = images[base + slide * unit: base + (slide + 1) * unit, ...]
        label = labels[base + slide * unit: (slide + 1) * unit, ...]
        if i in validate_subjects:
            if len(validate) == 0:
                validate = image
                validate_label = label
                continue
            validate = np.concatenate((validate, image), axis=0)
            validate_label = np.concatenate((validate_label, label), axis=0)
        elif i in test_subjects:
            if len(test) == 0:
                test = image
                test_label = label
                continue
            test = np.concatenate((test, image), axis=0)
            test_label = np.concatenate((test_label, label), axis=0)
        else:
            if len(train) == 0:
                train = image
                train_label = label
                continue
            train = np.concatenate((train, image), axis=0)
            train_label = np.concatenate((train_label, label), axis=0)
    
    train, validate, test = np.asarray(train), np.asarray(validate), np.asarray(test)
    train_label, validate_label, test_label = np.asarray(train_label), np.asarray(validate_label), np.asarray(test_label)
    print(train.shape)
    #data augmentation
    train = np.concatenate((train, train[..., ::-1, :]), axis=0)
    train_label = np.concatenate((train_label, train_label[..., ::-1, :]), axis=0)
    train_aug = np.zeros(train.shape, dtype=np.float32)
    label_aug = np.zeros(train_label.shape, dtype=np.float32)
    for i in range(train.shape[0]):
        if domain_knowledge and aging:
            train_aug[i, ..., 0], train_aug[i, ..., 1], train_aug[i, ..., 2], train_aug[i, ..., 3], label_aug[i, ..., 0] = \
                augmentation3(train[i, ..., 0], train[i, ..., 1], train[i, ..., 2], train[i, ..., 3], label[i, ..., 0])
        elif domain_knowledge or aging:
            train_aug[i, ..., 0], train_aug[i, ..., 1], train_aug[i, ..., 2], label_aug[i, ..., 0] = \
                augmentation2(train[i, ..., 0], train[i, ..., 1], train[i, ..., 2], label[i, ..., 0])
        else:
            train_aug[i, ..., 0], train_aug[i, ..., 1], label_aug[i, ..., 0] = \
                augmentation(train[i, ..., 0], train[i, ..., 1], label[i, ..., 0])

    train = np.concatenate((train, train_aug), axis=0)
    train_label = np.concatenate((train_label, label_aug), axis=0)

    train, validate, test = np.transpose(train, (0, 3, 1, 2)), np.transpose(validate, (0, 3, 1, 2)), np.transpose(test, (0, 3, 1, 2))
    train_label, validate_label, test_label = np.transpose(train_label, (0, 3, 1, 2)), np.transpose(validate_label, (0, 3, 1, 2)), np.transpose(test_label, (0, 3, 1, 2))

    return (train, train_label), (validate, validate_label), (test, test_label)

def Abvib_preprocessing(FLAIR_image):
    #print(np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
    FLAIR_image = FLAIR_image[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
    brain_mask_FLAIR = brain_mask_FLAIR[:, (image_rows_Dataset//2-rows_standard//2):(image_rows_Dataset//2+rows_standard//2), (image_cols_Dataset//2-cols_standard//2):(image_cols_Dataset//2+cols_standard//2)]
    ###------Gaussion Normalization here
    FLAIR_image = FLAIR_image - np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image = FLAIR_image / np.std(FLAIR_image[brain_mask_FLAIR == 1])
    #---------------------------------------------------
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    return FLAIR_image

if __name__ == "__main__":
    train, validate, test = split(validate_subjects=[20, 23], test_subjects = [59, 40], domain_knowledge=True, aging=True)
    print(train[0].shape, train[1].shape)
    print(validate[0].shape, validate[1].shape)
    print(test[0].shape, test[1].shape)
