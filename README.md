# BAGAU-Net

This repository contains implementation code for the paper "Brain Atlas Guided Attention U-Net for White Matter Hyperintensity Segmentation"

## Model Architecture

![alt text](model.jpg)

 Brain atlas guided attention U-Net (BAGAU-Net) that consists of two separate encoding-decoding paths. As shown in above, the upper path is a U-Net like architecture designed to extract semantic information from the image itself. The lower path is the atlas encoding path where the spatially registered atlas image is input to help guide the decoding process in the segmentation path. Moreover, we designed a multi-input attention module (MAM) and attention fusion module (AFM) to effectively combine the information between the two paths during the decoding process of segmentation path based on the attention gate (AG) as introduced in [attention U-Net](https://arxiv.org/abs/1804.03999).

 ## Environments

Use the following command to install required python packages:

 `pip install numpy==1.16.4 setuptools==41.0.1 tensorflow==1.14.0 torch==1.3.1 keras==2.0.5 SimpleITK==1.2.2 matplotlib scipy==1.3.0 scikit-learn==0.21.2 tqdm==4.32.1`

## Model Training

To run training, use the following command:

`bash run.sh`