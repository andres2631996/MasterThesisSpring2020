#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:32:23 2020

@author: andres
"""

import numpy as np

import os 

import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from patientStratification import StratKFold

import time

import torch # Deep learning package

from torch import nn

from torch.nn import functional as F

from torch import optim # Optimizer package

import torchvision

from torch.utils import data

import albumentations as A

from torch.autograd import Variable

import elasticdeform

import scipy.ndimage

import random

import math

import torchio

import SimpleITK as sitk

import warnings

from skimage.transform import resize, rescale

#from torchvision import datasets, transforms # Package to manipulate datasets



class QFlowDataset(data.Dataset):

    """
    Dataset for QFlow segmentation. 
    
    Params:
        
        - img_paths: folders with data
        
        - train: if the dataset is used for training (True) or for validation 
                 (False)
                 
        - work_with: type of images to work with ('mag'/'pha'/'magBF'/'both'/'bothBF')
        
        - threeD: if True, work with 2D+time data, else work with 2D slices
        
        - augmentation: if True, augment the data with some transformations,
        else return original dataset
        
        - probs: if augmentation is True, probabilities for different augmentation 
        events to happen
    
    """
    # img_paths is list of paths with images


    def __init__(self, img_paths, train, work_with, threeD, augmentation, probs):

        self.img_paths = img_paths
        
        self.train = train
        
        self.threeD = threeD
        
        self.work_with = work_with
        
        self.augmentation = augmentation
        
        self.probs = probs

        
#        
    def __len__(self):
        
        return len(self.img_paths)
    
    
    def readVTK(self, path, filename, order='F'):
            
        """
        Utility function to read vtk volume. 
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            - path: path where VTK file is located
            - filename: VTK file name
        
        Returns:
            
            - numpy array
            - data origin
            - data spacing
        
        """
    
        reader = vtk.vtkStructuredPointsReader()
    
        reader.SetFileName(path + filename)
    
        reader.Update()
    
        image = reader.GetOutput()
    
        numpy_array = vtk_to_numpy(image.GetPointData().GetScalars())
    
        numpy_array = numpy_array.reshape(image.GetDimensions(),order='F')
    
        numpy_array = numpy_array.swapaxes(0,1)
    
        origin = list(image.GetOrigin())
    
        spacing = list(image.GetSpacing())
    
        return numpy_array, origin, spacing

    
    
    
    
    def extractArrays(self, img_path):
        
        """
        Provide raw arrays and mask arrays of a certain path depending on the
        working method provided
        
        
        - Params:
            
            - inherited by the class (see class description)
            
            - path: folder with images where to look at 
        
        - Returns:
            
            - raw_arrays: list of raw arrays
            
            - mask_arrays: list of mask arrays
            
            - raw_names: list of raw array names (to check that raw files and masks correspond)
            
            - mask_arrays: list of mask array names (to check that raw files and masks correspond)
        
        
        """
        
        patient_files = sorted(os.listdir(img_path))
        
        raw_arrays = []
        
        mask_arrays = []
        
        raw_names = []
        
        mask_names = []
        
        if len(patient_files) == 0:
            
            print('Empty folder. Please specify an adequate folder')
        
        else:
            
            # Extract list with raw arrays
        
            if 'both' in self.work_with:
                
                if self.work_with == 'both' or self.work_with == 'Both' or self.work_with == 'BOTH':
                    
                    # Load both magnitude and phase images. Magnitude images are not bias-field-corrected
                    
                    ind_mag_files = [i for i, s in enumerate(patient_files) if 'mag_' in s]
                    
                    ind_pha_files = [i for i, s in enumerate(patient_files) if 'pha' in s]
                    
                    
                elif self.work_with == 'bothBF' or self.work_with == 'BothBF' or self.work_with == 'BOTHBF' or self.work_with == 'bothBf' or self.work_with == 'BothBf' or self.work_with == 'BOTHBf' or self.work_with == 'bothbf' or self.work_with == 'BOTHBF':
                    
                    # Load both magnitude and phase images. Magnitude images are bias-field-corrected
                    
                    ind_mag_files = [i for i, s in enumerate(patient_files) if 'magBF' in s]
                    
                    ind_pha_files = [i for i, s in enumerate(patient_files) if 'pha' in s]
                    
                cont_pha = 0
             
                for ind_mag in ind_mag_files:
                   
                    mag_array, _, _ = self.readVTK(img_path, patient_files[ind_mag])
                    
                    pha_array, _, _ = self.readVTK(img_path, patient_files[ind_pha_files[cont_pha]])
                    
                    final_array = np.zeros((mag_array.shape[0], mag_array.shape[1], mag_array.shape[2], 2))
                    
                    final_array[:,:,:,0] = mag_array
                    
                    final_array[:,:,:,1] = pha_array
                    
                    raw_names.append(patient_files[ind_mag])
                        
                    raw_arrays.append(final_array)
               
                    cont_pha += 1
            
            else:
            
                if self.work_with == 'mag' or self.work_with == 'Mag' or self.work_with == 'MAG':
        
                        
                    # Load only magnitude images without bias field correction
                   
                    ind_raw_files = [i for i, s in enumerate(patient_files) if 'mag_' in s]
               
               
                elif self.work_with == 'magBF' or self.work_with == 'MagBF' or self.work_with == 'MAGBF' or self.work_with == 'magBf' or self.work_with == 'MagBf' or self.work_with == 'MAGBf' or self.work_with == 'magbf' or self.work_with == 'MAGBF':
                   
                   # Load only magnitude images with bias field correction
                    
                   ind_raw_files = [i for i, s in enumerate(patient_files) if 'magBF' in s]
               
               
                elif self.work_with == 'pha' or self.work_with == 'Pha' or self.work_with == 'PHA':
                    
                    # Load only phase images
                   
                    ind_raw_files = [i for i, s in enumerate(patient_files) if 'pha' in s]
                    
             
                for ind_raw in ind_raw_files:
                    
                    raw_names.append(patient_files[ind_raw])
                   
                    raw_array, _, _ = self.readVTK(img_path, patient_files[ind_raw])
                        
                    raw_arrays.append(raw_array)
                   
                        
            
        # Extract list with mask arrays
        
        ind_mask_files = [i for i, s in enumerate(patient_files) if 'msk' in s]
       
        cont = 0
        
        for ind_mask in ind_mask_files:
            
            # Check that raw files and masks coincide in position in the dataset
            
            flag = 'continue'
            
            if 'both' in self.work_with:
                
                test_raw_name = patient_files[ind_mask].replace('msk', 'pha')
       
                if test_raw_name != patient_files[ind_pha_files[cont]]:
                    
                    flag = 'stop'
                    
                    print('Raw and mask files do not correspond')
            
            else:
                
                test_raw_name = patient_files[ind_mask].replace('msk', self.work_with)
         
                if test_raw_name != patient_files[ind_raw_files[cont]]:
                    
                    flag = 'stop'
                    
                    print('Raw and mask files do not correspond')
            
            
            if flag != 'stop':
            
                mask_array, _, _ = self.readVTK(img_path, patient_files[ind_mask])

                mask_names.append(patient_files[ind_mask])
                        
                mask_arrays.append(mask_array)
            
            cont += 1

        
        return raw_arrays, mask_arrays, raw_names, mask_names

    
        
    
    def zero_padding(self, mask, shape):
            
        """
        
        Zero pad mask image with shape of raw file.
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            
            - mask: 2D array to zero-pad

        
        Return: zero-padded mask in shape of raw file (result)
        
        """
        
        result = np.zeros(shape)
    
        center = np.array(np.array(shape)//2).astype(int) # Central voxel of raw file
        
        mask_half_shape = np.floor(np.array(mask.shape)/2).astype(int)
    
        if (np.remainder(mask.shape[0], 2) == 0) and (np.remainder(mask.shape[1], 2) == 0):
    
            result[center[0] - mask_half_shape[0] : center[0] + mask_half_shape[0], 
                   center[1] - mask_half_shape[1] : center[1] + mask_half_shape[1],:] = mask
    
        
        elif (np.remainder(mask.shape[0], 2) == 1) and (np.remainder(mask.shape[1], 2) == 0):
    
            result[center[0] - mask_half_shape[0] : center[0] + mask_half_shape[0] + 1, 
                   center[1] - mask_half_shape[1] : center[1] + mask_half_shape[1],:] = mask
                   
        elif (np.remainder(mask.shape[0], 2) == 0) and (np.remainder(mask.shape[1], 2) == 1):
    
            result[center[0] - mask_half_shape[0] : center[0] + mask_half_shape[0], 
                   center[1] - mask_half_shape[1] : center[1] + mask_half_shape[1] + 1,:] = mask
                   
        elif (np.remainder(mask.shape[0], 2) == 1) and (np.remainder(mask.shape[1], 2) == 1):
    
            result[center[0] - mask_half_shape[0] : center[0] + mask_half_shape[0] + 1, 
                   center[1] - mask_half_shape[1] : center[1] + mask_half_shape[1] + 1,:] = mask
               
        return result
    
    
    def executeTransform(self, img, seg, params):
        
        """
        Execute random affine transform on images for augmentation.
        
        Params:
            
            - img: 3D image to be transformed
            
            - seg: 3D segmentation to be transformed
            
            - params: affine transformation parameters
        
        
        Returns:
            
            - transformed_img: transformed 3D array
            
            - transformed_seg: transformed 3D segmentation
        
        
        """
        
        
        affine = sitk.AffineTransform(3)
        
#        affine.SetParameters((params[2]*np.cos(params[4]), -params[2]*np.sin(params[4]),0,
#                              params[3]*np.sin(params[4]), params[3]*np.cos(params[4]),0,
#                              0,0,1,
#                              params[0]*np.cos(params[4]) + params[1]*np.sin(params[4]), - params[0]*np.sin(params[4]) + params[1]*np.cos(params[4]), 0))
#        
        
        affine.SetParameters((params[2],np.cos(params[4]),0,
                              np.sin(params[4]), params[3],0,
                              0, 0, 1,
                              params[0], params[1], 0))
        
        resampler_img = sitk.ResampleImageFilter()
        
        resampler_seg = sitk.ResampleImageFilter()
        
        for i in range(img.shape[0]):
               
            if 'both' in self.work_with:
                
                for j in range(img.shape[-1]):
            
                    img_aux = sitk.GetImageFromArray(img[i,:,:,:,j])
                    
                    # Set the reference image
                    resampler_img.SetReferenceImage(img_aux)
                    
                    # Use a linear interpolator
                    resampler_img.SetInterpolator(sitk.sitkLinear)
                    
                    
                    # Set the desired transformation
                    resampler_img.SetTransform(affine)
                    
                    img_resampled = resampler_img.Execute(img_aux)
                    
                    transformed_img = sitk.GetArrayFromImage(img_resampled)
                    
                    img[i,:,:,:,j] = transformed_img
                
                seg_aux = sitk.GetImageFromArray(seg[i,:,:,:])
    
                resampler_seg.SetReferenceImage(seg_aux)
                
                resampler_seg.SetInterpolator(sitk.sitkLinear)
  
                resampler_seg.SetTransform(affine)

                seg_resampled = resampler_img.Execute(seg_aux)

                transformed_seg = sitk.GetArrayFromImage(seg_resampled)

                seg[i,:,:,:] = transformed_seg
                
            
            else:
                
                img_aux = sitk.GetImageFromArray(img[i,:,:,:], sitk.sitkFloat32)
                
                seg_aux = sitk.GetImageFromArray(seg[i,:,:,:], sitk.sitkFloat32)
    
                # Set the reference image
                resampler_img.SetReferenceImage(img_aux)
                
                resampler_seg.SetReferenceImage(seg_aux)
                    
                # Use a linear interpolator
                resampler_img.SetInterpolator(sitk.sitkLinear)
                
                resampler_seg.SetInterpolator(sitk.sitkLinear)
                        
                # Set the desired transformation
                resampler_img.SetTransform(affine)
                
                resampler_seg.SetTransform(affine)
                            
                img_resampled = resampler_img.Execute(img_aux)
                
                seg_resampled = resampler_img.Execute(seg_aux)
                
                transformed_img = sitk.GetArrayFromImage(img_resampled)
                
                transformed_seg = sitk.GetArrayFromImage(seg_resampled)
                
                seg[i,:,:,:] = transformed_seg
                
                img[i,:,:,:] = transformed_img
        
        return img, seg       
                
                
        
    def dataAugmentation(self, img, seg):
        
        """
        Perform data augmentation to images and segmentations with a series
        of probabilities.
        
        Params:
            
            - img: raw image to augment
            
            - seg: segmentation to augment
            
            - probs: probabilities for augmentation events to happen
            
            [translation, noise, rotation, flipping, elastic deformation]
        
        
        Return:
            
            - img: transformed raw image
            
            - seg: transformed segmentation
        
        """
#            
#       # Rescale image to half dimensions: faster augmentation
        
#        
#        # Add Gaussian noise
        
        t0 = time.time()
        
        r = random.uniform(0,1)
        
        if r < self.probs[0]:
            
            amp = random.uniform(0,0.15)
            
            std = amp*np.mean(img.flatten())
        
            noise = np.random.normal(scale = std, size = img.shape)
            
            img += noise
            
        
#        # Affine transforms in SimpleITK
#        
#        compare = [random.uniform(0,1)]*4
#        
#        if compare[0] < self.probs[1]:
#            
#            # Apply scaling
#            
#            scale_x = random.uniform(1,5)
#            
#            scale_y = random.uniform(1,5)
#        
#        else:
#            
#            scale_x = scale_y = 1
#            
#        
#        if compare[1] < self.probs[2]:
#            
#            # Apply translation
#            
#            if scale_x != 0 and scale_y != 0:
#                
#                D = (np.array(seg.shape)/(np.array([1,scale_x, scale_y,1]))//4).astype(int)
#                
#            else:    
#                
#                D = (np.array(seg.shape)//4).astype(int)
#            
#            shift_x = np.random.randint(low = 0, high = D[1])
#            
#            shift_y = np.random.randint(low = 0, high = D[2])
#        
#        else:
#            
#            shift_x = shift_y = 0
#        
#        if compare[3] < self.probs[3]:
#            
#            rot = np.random.randint(low = -30, high = 30)
#        
#        else:
#            
#            rot = 0
#        
#        params = [shift_x, shift_y, scale_x, scale_y, rot]
#        
#        img, seg = self.executeTransform(img, seg, params)
            
        
        # Random zooming: cropping + resizing
        
        #warnings.filterwarnings('ignore')
        
        
        r = random.uniform(0,1)
        
        t1 = time.time()
        
        print('Time for noise: {}'.format(t1 - t0))
        
        if r < self.probs[1]:
  
            zoom = random.uniform(1,1.2)
            
            crop_shape = (np.array([img.shape[1], img.shape[2]])/zoom).astype(int)
            
            center = (np.array([img.shape[1], img.shape[2]])//2).astype(int)
            
            if 'both' in self.work_with:
                
                orig_shape = img.shape
            
                img_cropped = img[:, (center[0] - crop_shape[0]//2):(center[0] + crop_shape[0]//2),(center[1] - crop_shape[1]//2):(center[1] + crop_shape[1]//2),:,:]
                
                img = resize(img_cropped, orig_shape, order = 0)
            
            else:
                
                orig_shape = img.shape
            
                img_cropped = img[:, (center[0] - crop_shape[0]//2):(center[0] + crop_shape[0]//2),(center[1] - crop_shape[1]//2):(center[1] + crop_shape[1]//2),:]
                
                img = resize(img_cropped, orig_shape, order = 0)
            
            orig_shape = seg.shape
            
            seg_cropped = seg[:, (center[0] - crop_shape[0]//2):(center[0] + crop_shape[0]//2),(center[1] - crop_shape[1]//2):(center[1] + crop_shape[1]//2),:]
                
            seg = resize(seg_cropped, orig_shape, order = 0)
         
        t2 = time.time()
        
        print('Time for zoom: {}'.format(t2-t1))
        
        # Random translation
            
        r = random.uniform(0,1)
        
        if r < self.probs[2]:
        
            D = (np.array(img.shape)//4).astype(int)
            
            shift_x = np.random.randint(low = 0, high = D[1])
            
            shift_y = np.random.randint(low = 0, high = D[2])
  
            if 'both' in self.work_with:
        
                img = scipy.ndimage.shift(img, (0, shift_x, shift_y, 0, 0), order = 0, mode = "nearest")
            
            else:
                
                img = scipy.ndimage.shift(img, (0, shift_x, shift_y, 0), order = 0, mode = "nearest")
            
            seg = scipy.ndimage.shift(seg, (0, shift_x, shift_y, 0), order = 0, mode = "nearest")
                
          
        t3 = time.time()
        
        print('Time for translation: {}'.format(t3-t2))
#         Add other transformations
#        
#        augmentation_pipeline = A.Compose(
#                [
#                    A.Flip(p = self.probs[1]),
#                    A.ShiftScaleRotate(p = self.probs[2])
#                                                
#                ],
#                p = 1
#            ) 
#        
#        img_aux = np.zeros(img.shape)
#        
#        seg_aux = np.zeros(seg.shape)
#        
#        for k in range(img.shape[3]):
#            
#            if 'both' in self.work_with:
#                
#                for i in range(img.shape[0]):
#                    
#                    for j in range(img.shape[-1]):
#        
#                        images_aug = augmentation_pipeline(image = img[i,:,:,k,j], mask = seg[i,:,:,k], axis = [(1, 2), (1, 2)])
#                        
#                        img_aux[i,:,:,k,j] = images_aug['image']
#        
#                        seg_aux[i,:,:,k] = images_aug['mask']
#            
#            else:
#                
#                for i in range(img.shape[0]):
#                    
#                    images_aug = augmentation_pipeline(image = img[i,:,:,k], mask = seg[i,:,:,k], axis = [(1, 2), (1, 2)])
#                        
#                    img_aux[i,:,:,k] = images_aug['image']
#    
#                    seg_aux[i,:,:,k] = images_aug['mask']
        
        # Random rotation
        
        
        r = random.uniform(0,1)
        
        if r < self.probs[3]:
            
            rot = random.uniform(-45, 45)
            
            for i in range(img.shape[0]):
                
                if 'both' in self.work_with:
                    
                    for k in range(img.shape[3]):
                    
                        for j in range(img.shape[-1]):
    
                            img[i,:,:,k,j] = scipy.ndimage.rotate(img[i,:,:,k,j], rot, order = 1, reshape = False)
                        
                        seg[i,:,:,k] = scipy.ndimage.rotate(seg[i,:,:,k], rot, order = 1, reshape = False)

            
                else: # Rotation in Y
                    
                    for k in range(img.shape[3]):
                    
                        img[i,:,:,k] = scipy.ndimage.rotate(img[i,:,:,k], rot, order = 1, reshape = False)
                        
                        seg[i,:,:,k] = scipy.ndimage.rotate(seg[i,:,:,k], rot, order = 1, reshape = False)
                   
        t4 = time.time()
        
        print('Time for rotation: {}'.format(t4-t3))

        
        return img, seg
        
#
    def __getitem__(self, index):
#
#        name = os.path.basename(self.img_paths[index])
#
#        # Load images depending on parameter "work_with"
        
        img_path = self.img_paths[index]
        
        raw_arrays, mask_arrays, raw_names, mask_names = self.extractArrays(img_path)
        
        img = np.asarray(raw_arrays)
        
        seg = np.asarray(mask_arrays)
    
        if self.train and self.augmentation:
            
            img, seg = self.dataAugmentation(img, seg)
            
        if not(self.threeD):
            
            if 'both' in self.work_with:
                
                img = img.reshape(img.shape[0]*img.shape[3], img.shape[1], img.shape[2], img.shape[4])
                
            else:
            
                img = img.reshape(img.shape[0]*img.shape[3], img.shape[1], img.shape[2])
            
            seg = seg.reshape(seg.shape[0]*seg.shape[3], seg.shape[1], seg.shape[2])
        
        # Transform list of raw files and mask files into tensor
        
        X = Variable(torch.from_numpy(img)).float()

        Y = Variable(torch.from_numpy(seg)).long()   
        
        print(img.shape, seg.shape, raw_names, mask_names)
    
        return X,Y, mask_names
        
        
            
#
#        seg_exists = len(self.seg_paths) > 0
#
#        # Load segmentation image if exists
#        if seg_exists:
#            seg_path = self.seg_paths[index]
#            seg = np.load(seg_path).astype("float32")#[0,:,:] #TODO remove ending
#        else:
#            seg = None
#        #seg = seg[:,:,0].astype("float32")



#
#
## Prepare data paths
#
#cont = 0
#
#for fold in patient_paths:
#    
#    patient_paths[cont] = [start_folder + path for path in fold]
#
#    cont += 1
#
## Lists with mask files and mask arrays for all folds
#
#files_folds = [] 
#
#mask_arrays_folds = []


# Lists with raw files and raw arrays for all folds

#if 'both' in train_with:
#        
#    mag_files_folds = []
#    
#    mag_arrays_folds = []
#    
#    pha_files_folds = []
#    
#    pha_arrays_folds = []
#
#
#else:
#    


#raw_arrays_folds = []
#        
#
#for fold in folds:
#    
#    # Lists with mask files and mask arrays found per fold
#    
#    files = []
#
#    mask_arrays = []
#    
#    # Take a look to parameter "train_with"
#    
#    if 'both' in train_with:
#        
#        mag_files = []
#        
#        mag_arrays = []
#        
#        pha_files = []
#        
#        pha_arrays = []
#    
#    
#    else:
#        
#        raw_files = []
#        
#        raw_arrays = []
#        
#    
#    # Loop through every fold
#    
#    for patient in fold:
#        
#        # Analyze patient by patient
#        
#        study_folder = '_' + patient[:4].lower() + '/'
#        
#        patient_folder = patient[5:] + '/'
#        
#        patient_path = start_folder + study_folder + patient_folder
#        
#        patient_files = sorted(os.listdir(patient_path))
#
#        ind_mask_files = [i for i, s in enumerate(patient_files) if 'msk' in s]
#        
#        for ind_mask in ind_mask_files:
#            
#            ind_ = [i for i, s in enumerate(patient_files[ind_mask]) if '_' in s]
#            
#            if 'venc' in patient_files[ind_mask]:
#                
#                patient_name = patient_files[ind_mask][:patient_files[ind_mask].index('msk')] + patient_files[ind_mask][ind_[-2]:]
#                
#            
#            else:
#            
#                patient_name = patient_files[ind_mask][:patient_files[ind_mask].index('msk')] + patient_files[ind_mask][ind_[-1]:]
#            
#            files.append(patient_name)
#            
#            mask_array, origin, spacing = readVTK(patient_path, patient_files[ind_mask])
#            
#            mask_arrays.append(mask_array)
#            
#            
#        
#        if not('both' in train_with):
#        
#            if train_with == 'mag' or train_with == 'Mag' or train_with == 'MAG':
#                
#                # Load only magnitude images without bias field correction
#               
#                ind_raw_files = [i for i, s in enumerate(patient_files) if 'mag_' in s]
#           
#           
#            elif train_with == 'magBF' or train_with == 'MagBF' or train_with == 'MAGBF' or train_with == 'magBf' or train_with == 'MagBf' or train_with == 'MAGBf' or train_with == 'magbf' or train_with == 'MAGBF':
#               
#               # Load only magnitude images with bias field correction
#                
#               ind_raw_files = [i for i, s in enumerate(patient_files) if 'magBF' in s]
#           
#           
#            elif train_with == 'pha' or train_with == 'Pha' or train_with == 'PHA':
#                
#                # Load only phase images
#               
#                ind_raw_files = [i for i, s in enumerate(patient_files) if 'pha' in s]
#                
#         
#            for ind_raw in ind_raw_files:
#               
#                raw_array, origin, spacing = readVTK(patient_path, patient_files[ind_raw])
#               
#                raw_arrays.append(raw_array)
#        
#        else:  
#        
#            if train_with == 'both' or train_with == 'Both' or train_with == 'BOTH':
#                
#                # Load both magnitude and phase images. Magnitude images are not bias-field-corrected
#                
#                ind_mag_files = [i for i, s in enumerate(patient_files) if 'mag_' in s]
#                
#                ind_pha_files = [i for i, s in enumerate(patient_files) if 'pha' in s]
#                
#                
#            elif train_with == 'bothBF' or train_with == 'BothBF' or train_with == 'BOTHBF' or train_with == 'bothBf' or train_with == 'BothBf' or train_with == 'BOTHBf' or train_with == 'bothbf' or train_with == 'BOTHBF':
#                
#                # Load both magnitude and phase images. Magnitude images are bias-field-corrected
#                
#                ind_mag_files = [i for i, s in enumerate(patient_files) if 'magBF' in s]
#                
#                ind_pha_files = [i for i, s in enumerate(patient_files) if 'pha' in s]
#                
#
#            cont_pha = 0
#         
#            for ind_mag in ind_mag_files:
#               
#                mag_array, origin, spacing = readVTK(patient_path, patient_files[ind_mag])
#                
#                pha_array, origin, spacing = readVTK(patient_path, patient_files[ind_pha_files[cont_pha]])
#                
#                final_array = np.zeros(mag_array.shape[0], mag_array.shape[1], mag_array.shape[2], 2)
#                
#                final_array[:,:,:,0] = mag_array
#                
#                final_array[:,:,:,1] = pha_array
#               
#                raw_arrays.append(final_array)
#           
#                cont_pha += 1
#
#        
##    if 'both' in train_with:
##    
##        mag_files_folds.append(mag_files)
##        
##        mag_arrays_folds.append(mag_arrays)
##        
##        pha_files_folds.append(pha_files)
##        
##        pha_arrays_folds.append(pha_arrays)
##        
##        #if fold == folds[-1]:
##            
##            #return mag_files_folds, mag_arrays_folds, pha_files_folds, pha_arrays_folds
##    
##    
##    else:
#        
#    files_folds.append(files)
#    
#    raw_arrays_folds.append(raw_arrays)
#    
#    mask_arrays_folds.append(mask_arrays)
    
    #if fold == folds[-1]:
        
        #return files_folds, raw_arrays_folds, mask_arrays_folds


#t2 = time.time()

#print('Data loading time: {} seconds'.format(t2 - t1))    