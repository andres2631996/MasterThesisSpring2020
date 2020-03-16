#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:32:23 2020

@author: andres
"""

import numpy as np

import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import torch # Deep learning package

from torch.utils import data

from torch.autograd import Variable

from augmentation import Augmentation # Augmentation for 2D + time

import params

import random

import augmentation2D # Augmentation for 2D






class QFlowDataset(data.Dataset):

    """
    Dataset for QFlow segmentation. 
    
    Params:
        
        - img_paths: raw data files 
        
        - mask_paths: ground truth files
        
        - train: if the dataset is used for training (True) or for validation or
                 testing (False)
        
        - augmentation: if True, augment the data with some transformations,
        else return original dataset
    
    """


    def __init__(self, img_paths, mask_paths, train, augmentation):

        self.img_paths = img_paths
        
        self.mask_paths = mask_paths
        
        self.train = train

        self.augmentation = augmentation
        
#        
    def __len__(self):
        
        return len(self.img_paths)
    
    
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


 
    def sumTime(self, path):

        """
        Provide sum of image along time as extra channel for 2D cases
        
        Params:
            
            - path: path of corresponding magnitude or phase file to analyze
        
        Returns:
            
            - sum_t: array with sum_t image array (2D array)

        """ 

        frame_ind = path.index('frame')           
        
        new_path = path[:frame_ind] + 'sum.vtk' # Sum time image path
        
        sum_t, _, _ = self.readVTK(new_path)
        
        return sum_t[:,:,0]
        
#
    def __getitem__(self, index):

        
        # Check that the files read coincide
        
        coincide = 1 # Coincidence flag
        
        raw_path = self.img_paths[index]
        
        if len(self.mask_paths) != 0:

            mask_path = self.mask_paths[index]
        
            # Make sure that raw and mask files are coincident
        
            if mask_path.replace('msk', 'pha') == raw_path or mask_path.replace('msk', 'mag') == raw_path or mask_path.replace('msk', 'magBF') == raw_path:
                
                coincide = 1
                
                mask_array, _, _ = self.readVTK(mask_path)
                
                if not(params.three_D):
                    
                    mask_array = mask_array[:,:,0]
                    
                    
                Y = Variable(torch.from_numpy(np.flip(mask_array,axis = 0).copy())).long()
            
            
            else:
                
                coincide = 0
                
                print('Raw files and ground truths are not correspondent\n')
                
                exit()

        
        if coincide == 1:
            
            # Compute number of channels
        
            if params.three_D:
                
                if 'both' in params.train_with:
                
                    channels = 2 # Magnitude + Phase
                
                else:
                    
                    channels = 1 # Only magnitude/only phase
            
            
            else:
                
                if 'both' in params.train_with:
                
                    channels = 4 # Magnitude + phase + sum of all magnitudes in time + sum of all phases in time
                
                else:
                    
                    channels = 2 # Magnitude/phase + sum of all magnitudes/phases in time

 
            raw,_,_ = self.readVTK(raw_path)
            
            if not(params.three_D): # 2D VTK files only have one slice in T
                
                img = np.zeros((raw.shape[0], raw.shape[1], channels))
            
            else:
            
                img = np.zeros((raw.shape[0], raw.shape[1], raw.shape[2],channels))
            
            if not('both' in params.train_with):
                
                if params.three_D:
                
                    img[:,:,:,0] = raw
                
                else:
                    
                    img[:,:,0] = raw[:,:,0]
                    
                    sum_t = self.sumTime(raw_path)
                    
                    img[:,:,1] = sum_t
    
            
            else:
                
                if 'BF' in params.train_with:
                
                    pha_path = raw_path.replace('magBF', 'pha')
                    
                else:
                    
                    pha_path = raw_path.replace('mag', 'pha')
    
                
                pha_array,_,_ = self.readVTK(pha_path)
                
                if params.three_D:
                
                    img[:,:,:,0] = raw
                    
                    img[:,:,:,1] = pha_array
                
                
                else:
    
                    img[:,:,0] = raw[:,:,0]
                    
                    img[:,:,1] = pha_array[:,:,0]
                    
                    img[:,:,2] = self.sumTime(raw_path)
                    
                    img[:,:,3] = self.sumTime(pha_path)
                    
            
            
            if self.train and self.augmentation:
                
                augm_chance = random.uniform(0,1)
                
                if augm_chance < params.augm_probs[0]:
    
                    if params.three_D:
                        
                        img = np.expand_dims(img, axis = 0)
                    
                        mask_array = np.expand_dims(mask_array, axis = 0)
                        
                        augm = Augmentation(img, mask_array)
                        
                        img, mask_array = augm.__main__()
                
                    
                    else:
                        
                        augm2D = augmentation2D.Augmentations2D(img, mask_array)
                        
                        img, mask_array = augm2D.__main__()
                        
        
        
            X = Variable(torch.from_numpy(np.flip(img,axis = 0).copy())).float()
        
            if not(params.three_D):
                
                X = X.permute(-1,0,1) # Channels first
        
            else:
                
                X = X.permute(-1,0,1,2) # Channels first
            
            
            if len(self.mask_paths) != 0:

                return X,Y, mask_path
            
            else:
                
                return X, raw_path
        

        
        