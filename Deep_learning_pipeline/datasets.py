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

import torch # Deep learning package

from torch.utils import data

from torch.autograd import Variable

import SimpleITK as sitk

from augmentation import Augmentation

import params

import matplotlib.pyplot as plt

import random

from torchvision.transforms import functional as tvF

import augmentation2D

#from torchvision import datasets, transforms # Package to manipulate datasets





class QFlowDataset(data.Dataset):

    """
    Dataset for QFlow segmentation. 
    
    Params:
        
        - img_paths: raw data files 
        
        - mask_paths: ground truth files
        
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


    def __init__(self, img_paths, mask_paths, train, work_with, threeD, augmentation):

        self.img_paths = img_paths
        
        self.mask_paths = mask_paths
        
        self.train = train
        
        self.threeD = threeD
        
        self.work_with = work_with
        
        self.augmentation = augmentation

        
#        
    def __len__(self):
        
        return len(self.mask_paths)
    
    
    def readVTK(self, filename, order='F'):
            
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
    
        reader.SetFileName(filename)
    
        reader.Update()
    
        image = reader.GetOutput()
    
        numpy_array = vtk_to_numpy(image.GetPointData().GetScalars())
    
        numpy_array = numpy_array.reshape(image.GetDimensions(),order='F')
    
        numpy_array = numpy_array.swapaxes(0,1)
    
        origin = list(image.GetOrigin())
    
        spacing = list(image.GetSpacing())
    
        return numpy_array, origin, spacing


 
                
    
        
#
    def __getitem__(self, index):

#
#        # Load images depending on parameter "work_with"
        
        # Check that the files read coincide
        
        coincide = 0 # Coincidence flag
        
        ind = index

        mask_path = self.mask_paths[ind]
        
        if len(self.img_paths) == 2:
            
            mag_paths = self.img_paths[0]
            
            pha_paths = self.img_paths[1]
            
            pha_path = pha_paths[ind]
            
            mag_path = mag_paths[ind]


            if (mask_path.replace('msk', 'mag') == mag_path or mask_path.replace('msk', 'magBF') == mag_path) and mask_path.replace('msk', 'pha') == pha_path:
                
                coincide = 1
            
            
            else:
                
                coincide = 0

        else:
            
            raw_paths = self.img_paths[0]
            
            raw_path = raw_paths[ind]
            
            if mask_path.replace('msk', 'pha') == raw_path or mask_path.replace('msk', 'mag') == raw_path or mask_path.replace('msk', 'magBF') == raw_path:
                
                coincide = 1
            
            
            else:
                
                coincide = 0
        
        if coincide == 1:
            
            mask_array,_, _ = self.readVTK(mask_path)
            
            if not(params.three_D): # 2D VTK files only have one slice in T
                
                mask_array = mask_array[:,:,0]
                
                img = np.zeros((mask_array.shape[0], mask_array.shape[1],len(self.img_paths)))
            
            else:
            
                img = np.zeros((mask_array.shape[0], mask_array.shape[1], mask_array.shape[2],len(self.img_paths)))
            
            if len(self.img_paths) == 1:
                
                raw_array,_,_ = self.readVTK(raw_path)
                
                if params.three_D:
                
                    img[:,:,:,0] = raw_array
                
                else:
                    
                    img[:,:,0] = raw_array[:,:,0]
            
            else:
                
                mag_array,_,_ = self.readVTK(mag_path)
                
                pha_array,_,_ = self.readVTK(pha_path)
                
                if params.three_D:
                
                    img[:,:,:,0] = mag_array
                    
                    img[:,:,:,1] = pha_array
                
                
                else:

                    img[:,:,0] = mag_array[:,:,0]
                    
                    img[:,:,1] = pha_array[:,:,0]
                    
            
        else:
            
            print('Raw files and ground truths are not correspondent\n')
            
            exit()

        
        # Compute number of channels and obtain sum of slices over time
        
        if 'both' in params.train_with:
            
            channels = 4
            
            
        
        else:
            
            channels = 2

        
        
            
        sum_time = np.sum(img, axis = 2)/img.shape[2]
    
        if self.train and self.augmentation and self.threeD:
            
            img = np.expand_dims(img, axis = 0)
        
            mask_array = np.expand_dims(mask_array, axis = 0)
            
            augm = Augmentation(img, mask_array, params.augm_params, params.augm_probs, self.work_with)
            
            img, mask_array = augm.__main__()

        
        elif self.train and self.augmentation and not(self.threeD):
            
            augm2D = augmentation2D.Augmentations2D(img, mask_array)
            
            img, mask_array, sum_time = augm2D.__main__()

        
        
        elif (not(self.augmentation) and not(self.threeD)) or (not(self.train) and not(self.threeD)):
            
            random_slice = np.random.randint(low = 0, high = img.shape[2]) # Pick a 2D slice from the volume at random
            
            img = img[:,:,random_slice,:]
            
            mask_array = mask_array[:,:,random_slice]
        
            

        if not(self.threeD):
            
            aux_img = np.zeros((img.shape[0], img.shape[1], channels))
            
            
            
            if channels == 2:
                
                if len(img.shape) == 2:
            
                    aux_img[:,:,:1] = np.expand_dims(img,-1)
                    
                    aux_img[:,:,1:] = np.expand_dims(sum_time,-1)
                
                else:
                    
                    aux_img[:,:,:1] = img
                    
                    aux_img[:,:,1:] = sum_time
            
            elif channels == 4:
                
                aux_img[:,:,:channels//2] = img
            
                aux_img[:,:,channels//2:] = sum_time
            
            # Transform list of raw files and mask files into tensor
            
            # Add extra channel with sum over time for image tensor
            
            # Tensor with regular image data
            
            X = Variable(torch.from_numpy(np.flip(aux_img,axis = 0).copy())).float()
            
            X = X.permute(-1,0,1) # Channels first

            Y = Variable(torch.from_numpy(np.flip(mask_array,axis = 0).copy())).long()
            
        else:
            
            X = Variable(torch.from_numpy(np.flip(img,axis = 0).copy())).float()
            
            X = X.permute(-1,0,1,2) # Channels first
    
            Y = Variable(torch.from_numpy(np.flip(mask_array,axis = 0).copy())).long()
        
         

        return X,Y, mask_path
        

        
        