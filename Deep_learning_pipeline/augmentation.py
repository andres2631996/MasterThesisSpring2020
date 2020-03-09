# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:16:16 2020

@author: andre
"""

import scipy.ndimage

import numpy as np

import random

import matplotlib.pyplot as plt

from torchvision import transforms

import torchvision.transforms.functional as TF

from PIL import Image


class RRC(transforms.RandomAffine):
    
    def __call__(self, img, mask):
        
        """
        Args:
            img: 3D array

        Returns:
            PIL Image: Randomly cropped and resized image.
            
        """
        
        im = Image.fromarray(img[:,:,img.shape[-1]//2])
        
        i, j, h, w = self.get_params(im, self.scale, self.ratio)
        
        img_final = np.zeros(img.shape)
        
        mask_final = np.zeros(mask.shape)

        for i in range(img.shape[-1]):
            
            im = Image.fromarray(img[:,:,i])

            trans_im = transforms.RandomAffine(img[:,:,i], i, j, h, w, self.size, self.interpolation)
            
            trans_mask = transforms.RandomAffine(mask[:,:,i], i, j, h, w, self.size, self.interpolation)
            
            img_final[:,:,i] = np.array(trans_im)
            
            mask_final[:,:,i] = np.array(trans_mask)

        return img_final, mask_final
    
    

class Augmentation:
    
    """
    Perform augmentation on given images: Gaussian noise, translation, rotation,
    scaling, horizontal/vertical flipping, sequence reverse (time flipping).
    
    Params:
        
        - img: image to augment (N, H, W, T, channels)
        
        - seg: corresponding mask to augment
        
        - limit_params: limit parameters to deal with
        
        - probs: probabilities for events to happen
        
        - work_with: type of images that are being augmented
    
    
    """
    
    def __init__(self, img, seg, limit_params, probs, work_with):
        
        self.img = img
        
        self.seg = seg
        
        self.limit_params = limit_params
        
        self.probs = probs
        
        self.work_with = work_with
    
    
    

    def parameterRandomization(self):
        
        """
        Randomize parameters for scale, translation, rotation and flipping
        
        Params:
            
            - img: image where to apply later transforms
            
            - probs: probabilities for each event
            
            - limit_params: limit parameters for each event
            
            
        Returns: [scale, trans_x, trans_y, angle, flip(1 or 2, 1: horizontal, 2: vertical), time_flip]
        
        """
        
        r_noise = random.uniform(0,1)
        
        r_sc = random.uniform(0,1)
        
        r_rot = random.uniform(0,1)
        
        r_flip = random.uniform(0,1)
        
        r_flip_temp = random.uniform(0,1)
        
        # Noise randomization
        
        if r_noise < self.probs[0]:
            
            amp = random.uniform(0,self.limit_params[0])
        
        else:
            
            amp = 0
        
        # Scale randomization
        
        if r_sc < self.probs[1]:
            
            sc = random.uniform(0,self.limit_params[1])
    
            scale = 1 + (np.random.rand()-1) * sc
            
        else:
    
            scale = 1
        
            
        
        # Rotation randomization
        
        if r_rot < self.probs[2]:
            
            angle = random.uniform(-self.limit_params[2], self.limit_params[2])*np.pi/180
    
        else:
            
            angle = 0
        
        
        # Flipping randomization
        
        if r_flip < self.probs[3]:
            
            flip = np.random.randint(1,3)
        
        else:
            
            flip = 0
    
        
        # Temporal flipping randomization
        
        if r_flip_temp < self.probs[4]:
            
            flip_temp = 1
        
        else:
            
            flip_temp = 0
    
    
        out_params = [amp, scale, angle, flip, flip_temp]
        
        return out_params



    def __main__(self):
        
        """
        Augment given image with a set of parameters
        
        - Params:
            
            img: image to augment
            
            seg: mask to augment
            
            limit_params: set of parameters of augmentation events
            
            probs: probabilities of augmentation events
        
        
        - Outputs:
            
            final_img: augmented image
            
            final_seg: augmented mask
        
        """
        
        # Join image and mask into same nd-array
            
        inp = np.zeros((self.img.shape[0], self.img.shape[1], self.img.shape[2], self.img.shape[3], self.img.shape[4] + 1))
        
        inp[:,:,:,:,0:-1] = self.img
        
        inp[:,:,:,:,-1] = self.seg
        
        trans = np.zeros(inp.shape)
        

        # Affine transformation parameters
        
        out_params = self.parameterRandomization()
        
        transform = np.array([[out_params[1]*np.cos(out_params[2]),-out_params[1]*np.sin(out_params[2])],
                             [out_params[1]*np.sin(out_params[2]),out_params[1]*np.cos(out_params[2])]])
        
        c_in = 0.5*np.array((self.img.shape[1], self.img.shape[2]))
        
        c_out = np.array((self.img.shape[1]/2, self.img.shape[2]/2))
        
        offset = c_in-c_out.dot(transform)
        
        for k in range(inp.shape[0]):
    
            for j in range(inp.shape[-1]):

                # Affine transformations
            
                for i in range(inp.shape[3]):
                    
                    # Scaling/Rotation/Translation
                    
                    trans[k,:,:,i,j] = scipy.ndimage.interpolation.affine_transform(
                        inp[k,:,:,i,j],transform.T,order = 0, offset = offset, 
                        output_shape=(self.img.shape[1],self.img.shape[2]),
                        cval=0.0,output=np.float32)
    
                    
        if out_params[-2] > 0: # Flipping

            if out_params[-2] == 1: # Horizontal flipping
                
                trans = np.flip(trans, axis = 2)
        
            elif out_params[-2] == 2: # Horizontal flipping
                
                trans = np.flip(trans, axis = 1)             
                    
        # Temporal flipping
                        
        if out_params[-1] == 1:
            
            trans = np.flip(trans, axis = 3) 
        
        # Separate image and mask and squeeze them
        
        final_img = trans[:,:,:,:,0:-1]
        
        final_seg = trans[:,:,:,:,-1]
        
        final_img = np.squeeze(final_img)
        
        final_seg = np.squeeze(final_seg)
            
            
        return final_img, final_seg


    


# if __name__ == "__main__":  