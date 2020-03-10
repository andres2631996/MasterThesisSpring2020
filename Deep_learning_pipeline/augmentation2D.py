#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:35:02 2020

@author: andres
"""

import numpy as np

import matplotlib.pyplot as plt

import albumentations as A

import os

import params

import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import time

import cv2



class Augmentations2D:
    
    def __init__(self, img, mask):
        
        self.img = img
        
        self.mask = mask
        
        
    def augment_slices(self,aug, data):
        
        
        images_aug = aug(**data)

        img, mask_aux = images_aug["image"], images_aug["mask"]
        
        mask = mask_aux[:,:,0]
        
        sum_t = mask_aux[:,:,1:]
        
        
        if len(img.shape) != 3:
            
            img = np.expand_dims(img, axis = -1)
            
        if len(sum_t.shape) != 3:
            
            sum_t = np.expand_dims(sum_t, axis = -1)
        
        return img, mask, sum_t
    

    
    
    def __main__(self):
        
        # Define set of transformations
        
        augmentation_pipeline = A.Compose(
                    [
                        A.ShiftScaleRotate(scale_limit = (params.augm2D_limits[0], 1), p = params.augm2D_probs[1], 
                                           rotate_limit = params.augm2D_limits[1], interpolation = 0, border_mode = cv2.BORDER_CONSTANT),
                        A.Flip(p = params.augm2D_probs[2]) # random flipping                             
                    ],
                    p = params.augm2D_probs[0]
                )
        
        
    
        
        mask_new = np.zeros((self.mask.shape[0], self.mask.shape[1], 1 + self.sum_t.shape[2]))
        
        mask_new[:,:,0] = self.mask
        
        mask_new[:,:,1:] = self.sum_t
        
        data = {"image": self.img, "mask": mask_new}

        
        img, mask, sum_t = self.augment_slices(augmentation_pipeline, data)

        
        return img, mask, sum_t
    

def readVTK(filename, order='F'):
            
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



#test_file = 'CKD2_CKD021_MRI3_dx_mag_0.vtk'
#
#mask_file = 'CKD2_CKD021_MRI3_dx_msk_0.vtk'
#
#img,_,_ = readVTK(test_file)
#
#mask,_,_ = readVTK(mask_file)
#
#img = np.expand_dims(img,axis = -1)
#
#t1 = time.time()
#
#augm2d = Augmentations2D(img, mask)
#
#
#img_trans, mask_trans, sum_trans = augm2d.__main__()
#
#t2 = time.time()
#
#print(t2-t1)
#
#plt.figure(figsize = (13,5))
#
#plt.subplot(141)
#
#plt.imshow(img_trans[:,:,0],cmap = 'gray')
#
#plt.subplot(142)
#
#plt.imshow(mask_trans,cmap = 'gray')
#
#plt.subplot(143)
#
#plt.imshow(sum_trans[:,:,0],cmap = 'gray')
#
#plt.subplot(144)
#
#plt.imshow(img_trans[:,:,0]*mask_trans,cmap = 'gray')