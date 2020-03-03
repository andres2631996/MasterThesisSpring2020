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


    def __init__(self, img_paths, train, test, work_with, threeD, augmentation):

        self.img_paths = img_paths
        
        self.train = train
        
        self.test = test
        
        self.threeD = threeD
        
        self.work_with = work_with
        
        self.augmentation = augmentation

        
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

        raw_names = []

        if not(self.test):
            
            mask_arrays = []
            
            mask_names = []
        
        if len(patient_files) == 0:
            
            print('Empty folder. Please specify an adequate folder\n')
        
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
        
        if not(self.test):
        
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
        
        
        else:
            
            return raw_arrays, raw_names

    
      
                
    
        
#
    def __getitem__(self, index):
#
#        name = os.path.basename(self.img_paths[index])
#
#        # Load images depending on parameter "work_with"
        
        img_path = self.img_paths[index]
        
        if not(self.test):
        
            raw_arrays, mask_arrays, raw_names, mask_names = self.extractArrays(img_path)
            
            seg = np.asarray(mask_arrays)
        
        else:
        
            raw_arrays, raw_names = self.extractArrays(img_path)
        
        img = np.asarray(raw_arrays)
        
    
        if self.train and self.augmentation:
            
            augm = Augmentation(img, seg, params.augm_params, params.augm_probs, self.work_with)
            
            img, seg = augm.__main__()

            
        if not(self.threeD):
            
            if 'both' in self.work_with:
                
                img = img.reshape(img.shape[0]*img.shape[3], img.shape[1], img.shape[2], img.shape[4])
                
            else:
            
                img = img.reshape(img.shape[0]*img.shape[3], img.shape[1], img.shape[2])
            
            if not(self.test):
            
                seg = seg.reshape(seg.shape[0]*seg.shape[3], seg.shape[1], seg.shape[2])


        # Transform list of raw files and mask files into tensor
        
        X = Variable(torch.from_numpy(np.flip(img,axis=0).copy())).float()
        
        if not(self.test):

            Y = Variable(torch.from_numpy(np.flip(seg,axis=0).copy())).long()   

            return X,Y, mask_names
        
        else:
            
            return X, raw_names
        
        