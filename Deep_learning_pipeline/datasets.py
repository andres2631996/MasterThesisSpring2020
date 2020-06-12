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

import matplotlib.pyplot as plt

import os




class QFlowDataset(data.Dataset):

    """
    Dataset for QFlow/2D PC-MRI images. 
    
    Params:
        
        - img_paths: raw data files (list of str)
        
        - mask_paths: ground truth files (list of str)
        
        - train: if the dataset is used for training (True) or for validation or
                 testing (False) (bool)
        
        - augmentation: if True, augment the data with some transformations,
        else return original dataset (bool)
    
    """


    def __init__(self, img_paths, mask_paths, train, augmentation):

        self.img_paths = img_paths
        
        self.mask_paths = mask_paths
        
        self.train = train

        self.augmentation = augmentation
        
#        
    def __len__(self): # Define length of dataset: as many image paths as we input
        
        if params.k > 1 or params.test:
        
            return len(self.img_paths)
        
        else:
            
            return len(self.img_paths[0])
    
    
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
        Provide sum of image along time and MIP along time as extra channel for 2D cases
        
        Params:
        
            - inherited by class (see description in the beginning)
            
            - path: path of corresponding magnitude or phase file to analyze (str)
        
        Returns:
            
            - sum_t: array with sum image array (2D array)
            
            - mip: array with MIP along time (2D array)

        """
        
        frame_ind = path.index('frame') 
        
        if not(params.multi_view) or (not('coronal' in path) and not('sagittal' in path)):
            
            # In 2D cases, sums along time and MIPs along time were saved as extra VTK files, just access them

            sum_path = path[:frame_ind] + 'sum.vtk' # Sum time image path

            mip_path = path[:frame_ind] + 'mip.vtk' # MIP image path
   
            
        else:
            
            if 'coronal' in path: # Multi-view case (unused)
                
                sum_path = path[:frame_ind] + 'coronal_sum.vtk'
                
                mip_path = path[:frame_ind] + 'coronal_mip.vtk'
                
            elif 'sagittal' in path: # Multi-view case (unused)
                
                sum_path = path[:frame_ind] + 'sagittal_sum.vtk'
                
                mip_path = path[:frame_ind] + 'sagittal_mip.vtk'
                
        sum_t, _, _ = self.readVTK(sum_path)

        mip,_,_ = self.readVTK(mip_path)
        
        return sum_t[:,:,0], mip[:,:,0]
    
    
    
    def addNeighbors(self, path, central_array):
        
        """
        Create a 2D+time array with given past and future neighbors from 
        "add3d" parameter in "params.py"
        
        Params:
        
            - path: path of central slice (str)
            
            - central_array: 2D central array (2D array)
            
        Returns:
        
            - final_array: output 2D+time array with past and future neighbors (3D array)
        
        """
        
        
        final_array = np.zeros((central_array.shape[0], central_array.shape[1], 2*params.add3d + 1))
                        
        final_array[:,:,params.add3d] = central_array

        # Find all slice files to be considered neighboring slices

        reversed_path = path[::-1]

        ind = -(reversed_path.index('/') + 1)

        slices_folder = sorted(os.listdir(path[:ind + 1])) # All slice files found in the same folder

        all_slices = sorted([s for s in slices_folder if ((path[(ind + 1):(-7)] in s) and (not('mip' in s)))]) # All slice filenames of the same volume
        
        all_slices = [item for item in all_slices if not('sum' in item) and not('mip' in item)]

        num_slices = len(all_slices)

        # Get slice index of central slice

        key_str = path[-6:]

        if key_str[0] == 'e':

            slice_ind = int(path[-5])

        else:

            slice_ind = int(key_str[:2])

        cont = 0

        for n in range(params.add3d, 0, -1):

            # Build dataset with neighboring slices

            # Get past index

            if slice_ind - n < 0:

                ind_past = num_slices - n

            else:

                ind_past = slice_ind - n

            # Get future index

            if slice_ind + n > num_slices - 1:

                ind_future = slice_ind + n - num_slices

            else:

                ind_future = slice_ind + n

            past_file = path.replace(str(slice_ind) + '.vtk', str(ind_past) + '.vtk')

            future_file = path.replace(str(slice_ind) + '.vtk', str(ind_future) + '.vtk')

            past_array,_,_ = self.readVTK(past_file)

            future_array,_,_ = self.readVTK(future_file)

            final_array[:,:,cont] = past_array[:,:,0]

            final_array[:,:,-cont-1] = future_array[:,:,0]

            cont += 1
            
        return final_array
    
    
    def othExtractor(self, raw_path, raw_array):
        
        """
        Extracts the corresponding 'OTH' image in case we want to train with 'OTH' images as extra channels
        
        OTH is a third modality apart from magnitude and phase that is found on QFLOW images from Philips and Siemens vendors
        
        (OTH comes "other")
        
        (if '_oth' in params.train_with)
        
        """
   
        # Get primary images with which the training is being done (magnitude + oth, phase + oth, or magnitude + phase + oth)
    
        splitting = params.train_with.split('_')
        
        primary = splitting[0]
        
        if primary == 'both' or primary == 'mag':
                
            oth_path = raw_path.replace('mag','oth')

        elif primary == 'magBF' or primary == 'bothBF':

            oth_path = raw_path.replace('magBF','oth')

        elif primary == 'pha':

            oth_path = raw_path.replace('pha','oth')
        
        if 'mip' in params.train_with: # Take the MIP of the OTH files (only available in 2D mode)
            
            ind_frame = oth_path.index('frame')
            
            oth_path = oth_path[:ind_frame] + 'mip.vtk'
            
        if os.path.exists(oth_path): # Existing modality (Philips, Siemens)
             
            oth_array, origin, spacing = self.readVTK(oth_path)
            
            ind = np.where(oth_array == oth_array[0,0,0])
            
            oth_array[ind] = -1 # Set all background values of OTH to -1

            if not(params.three_D):

                oth_array = oth_array[:,:,0]

            if params.add3d > 0:
                
                if not('mip' in oth_path):

                    oth_array = self.addNeighbors(oth_path, oth_array) # Provide a 2D+time array with OTH modality with past and future frames
                    
                else: # Repeat MIP channel along time in 2D+time cases
                    
                    aux = np.repeat(oth_array, params.add3d*2+1, -1)
                    
                    oth_array = np.reshape(aux, (oth_array.shape[0], oth_array.shape[1], params.add3d*2+1))
                
        else: # Non-existing OTH modality (GE images)
            
            #oth_array = np.zeros(raw_array.shape) + 0.5
            oth_array = np.ones(raw_array.shape)
            
        return oth_array
                
            
#
    def __getitem__(self, index):

        
        # Check that the image and mask files that are read coincide
        
        coincide = 1 # Coincidence flag
        
        if params.k > 1 or params.test:
        
            raw_path = self.img_paths[index]
            
        else:
            
            raw_path = self.img_paths[0][index]
        
        if len(self.mask_paths) != 0:
            
            if params.k > 1 or params.test:

                mask_path = self.mask_paths[index]
                
            else:
                
                mask_path = self.mask_paths[0][index]
        
        
            if mask_path.replace('msk', 'pha') == raw_path or mask_path.replace('msk', 'mag') == raw_path or mask_path.replace('msk', 'magBF') == raw_path:
                
                coincide = 1
                
                mask_array, _, _ = self.readVTK(mask_path) # Files coincide, read mask
                
                if not(params.three_D): # In case of 2D models
                    
                    mask_array = mask_array[:,:,0]
                    
                    if params.add3d > 0: # In case of 2D+time models built from 2D images with past and future neighbors
                        
                        mask_array = self.addNeighbors(mask_path, mask_array) # Surround central slice with past and future neighbors
            
            else:
                
                coincide = 0 # Non-correspondent images and ground-truths
                
                print('Raw files and ground truths are not correspondent\n')
                
                exit()

        
        if coincide == 1:
            
            # Compute number of channels
        
            if params.three_D or (not params.three_D and not(params.sum_work)): # No extra priors are used in 2D, or 2D+time is used
                
                if 'both' in params.train_with:
                
                    channels = 2 # Magnitude + Phase
                
                else:
                    
                    channels = 1 # Only magnitude/only phase

            
            
            else: # 2D case with extra prior information (sum, MIP along time)
                
                if 'both' in params.train_with:
                
                    channels = 7 # Magnitude + phase + sum of all magnitudes in time + sum of all phases in time + MIP phase + MIP magnitude + Magnitude*Phase
                
                else:
                    
                    channels = 3 # Magnitude/phase + sum of all magnitudes/phases in time + MIP of magnitude/phase along time
                
            if '_oth' in params.train_with:

                channels += 1 # Existing channels + 'oth' modality
                
 
            raw,_,_ = self.readVTK(raw_path) # Read image
    
            if not(params.three_D):
            
                raw = raw[:,:,0]
                
                
    
            if params.add3d > 0:
            
                raw = self.addNeighbors(raw_path, raw) # Surround central slice with past and future neighbors
                    
            
            if not(params.three_D) and params.add3d == 0: # 2D VTK files only have one slice in T
                
                img = np.zeros((raw.shape[0], raw.shape[1], channels))
            
            elif params.three_D and params.add3d == 0: # Full 2D+time volumes (unused)
            
                img = np.zeros((raw.shape[0], raw.shape[1], raw.shape[2],channels))
            
            elif not(params.three_D) and params.add3d > 0: # 2D+time volumes with neighboring past and future frames
                
                img = np.zeros((raw.shape[0], raw.shape[1], params.add3d*2 + 1,channels))
                
                
                
            
            if not('both' in params.train_with): # Images only contain magnitude or phase information
                
                if params.three_D or (not(params.three_D) and params.add3d > 0): # 2D+time case
                
                    img[:,:,:,0] = raw  # First channel: magnitude/phase
                
                else: # 2D case
                    
                    img[:,:,0] = raw # First channel: magnitude/phase
                    
                    if params.sum_work: # Work with extra channel with sum of time frames
                    
                        sum_t, mip = self.sumTime(raw_path)

                        img[:,:,1] = sum_t # Second channel: sum along time
                        
                        img[:,:,2] = mip # Third channel: MIP along time
            
            else: # Training with both magnitude and phase images
                
                if 'BF' in params.train_with: # Training with bias-field corrected magnitude images (bias field = BF)
                
                    pha_path = raw_path.replace('magBF', 'pha')
                    
                else:
                    
                    pha_path = raw_path.replace('mag', 'pha')
    
                
                pha_array,_,_ = self.readVTK(pha_path) # Read phase image
            
                if params.add3d > 0: 
                    
                    pha_array = self.addNeighbors(pha_path, pha_array[:,:,0]) # Surround central slice with past and future neighbors
                    
                
                if params.three_D or (not(params.three_D) and params.add3d > 0): # 2D+time case
                
                    img[:,:,:,0] = raw # First channel: magnitude image
                    
                    img[:,:,:,1] = pha_array # Second channel: phase image
                
                
                else: # 2D case
    
                    img[:,:,0] = raw # First channel: magnitude image

                    if params.sum_work: # Work with extra channels with sums and MIPs of time frames
                    
                        img[:,:,1], img[:,:,2] = self.sumTime(raw_path) # Second channel: sum along time magnitude // Third channel: MIP along time magnitude
                        
                        img[:,:,3] = pha_array[:,:,0] # Fourth channel: phase image

                        img[:,:,4], img[:,:,5] = self.sumTime(pha_path) # Fifth channel: sum along time phase // Sixth channel: MIP along time phase
                        
                        img[:,:,6] = img[:,:,0]*img[:,:,3] # Seventh channel: magnitude * phase
                        
                    else: # Work only with magnitude and phase
                        
                        img[:,:,1] = pha_array[:,:,0] # Second channel: phase image
                        
                    
            if '_oth' in params.train_with: # Incorporate extra modality OTH as last channel
    
                oth_array = self.othExtractor(raw_path, raw)

                if not(params.three_D) and params.add3d == 0:

                    #img[:,:,0] *= oth_array 
                    
                    img[:,:,-1] = oth_array 

                else:

                    #img[:,:,:,0] *= oth_array
                    
                    img[:,:,:,-1] = oth_array
                    
            
            if self.train and self.augmentation: # Perform augmentation of the training set
                
                augm_chance = random.uniform(0,1) # Overall augmentation probability
                
                if augm_chance < params.augm_probs[0]: # Perform augmentation
    
                    #if params.three_D or (not(params.three_D) and params.add3d > 0): # (3D augmentation --> UNUSED)
                        
                     #   img = np.expand_dims(img, axis = 0)
                    
                      #  mask_array = np.expand_dims(mask_array, axis = 0)
                        
                      #  augm = Augmentation(img, mask_array)
                        
                      #  img, mask_array = augm.__main__()
                        
                       # if not('both' in params.train_with):
                            
                        #    img = img[0,:,:,:,:]
                            
                         #   mask_array = mask_array[0,:,:,:]

                        
                    
                    #else:
                    
                        if params.three_D or (not(params.three_D) and params.add3d > 0): # To augment a 2D+time image as 2D, move all frames in all channels to the channels dimension
                            
                            img = img.reshape((img.shape[0], img.shape[1], img.shape[-2]*img.shape[-1]))

                        augm2D = augmentation2D.Augmentations2D(img, mask_array) # Augmentation in Albumentations

                        img, mask_array = augm2D.__main__()
                        
                        if params.three_D or (not(params.three_D) and params.add3d > 0): # In case of 2D+time images, move the time frames again to an extra dimension
                            
                            img = img.reshape((img.shape[0], img.shape[1], img.shape[-1]//channels, channels))

        
            X = Variable(torch.from_numpy(np.flip(img,axis = 0).copy())).float() # Convert model input to PyTorch tensor and flip it
            
            if len(self.mask_paths) != 0:

                Y = Variable(torch.from_numpy(np.flip(mask_array,axis = 0).copy())).long() # Convert model ground-truth into PyTorch tensor to flip it
            
            else:
                
                Y = []
        
            if params.three_D or (not(params.three_D) and params.add3d > 0):
                
                X = X.permute(-1,0,1,2) # Channels first
                
        
            else:
                
                X = X.permute(-1,0,1) # Channels first
            

                
            return X, Y, raw_path
        

        
        