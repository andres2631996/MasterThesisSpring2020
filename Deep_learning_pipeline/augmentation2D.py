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

import time

import cv2

from PIL import Image,ImageEnhance

import random



class Augmentations2D:
    
    """
    Augment 2D arrays as (B,C,H,W) with Albumentations library.
    
    Optionally allows for augmentation of 2D+time arrays, expressing time frames as separate channels of a 2D array, and then rewriting the array in 2D+time after augmentation
    
    Params:
    
    - img: training image to augment (array)
    
    - mask: training ground-truth to augment identically as the image (array)
    
    """
    
    def __init__(self, img, mask):
        
        self.img = img
        
        self.mask = mask

        
        
    def augment_slices(self,aug, data):
        
        """
        Module where transformations are applied.
        
        Params:
        
            - aug: augmentation parameters (Albumentations Compose transformation)
            
            - data: data to be transformed (dictionary: {'image': [], 'mask': []})
            
        Return:
        
            - img: augmented image (array)
            
            - mask_aux: augmented mask (array)
        
        
        """
        
        
        images_aug = aug(**data)

        img, mask_aux = images_aug["image"], images_aug["mask"]
            
        if len(img.shape) != 3:

            img = np.expand_dims(img, axis = -1)


        return img, mask_aux
            

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
            
        data = {"image": self.img, "mask": self.mask}

        img_final, mask = self.augment_slices(augmentation_pipeline, data) # Transformation application
        
        if params.three_D or (not(params.three_D) and params.add3d > 0):
            
            # Time flipping: different from the Albumentations flipping, which were horizontal or vertical
            
            if random.uniform(0,1) < params.augm2D_probs[3]:
            
                img_final = np.flip(img_final, axis = -1)

        
        return img_final, mask
    

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