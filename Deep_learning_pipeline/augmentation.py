# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:16:16 2020

@author: andre
"""

import scipy.ndimage

import numpy as np

import random

import sys

import params
    
    

class Augmentation:
    
    """
    Perform augmentation on given images: Gaussian noise, translation, rotation,
    scaling, horizontal/vertical flipping, sequence reverse (time flipping).
    
    It is just used in 2D+time, since it allows to augment neighboring frames in
    the same way.
    
    Params:
        
        - img: image to augment (N, H, W, T, channels)
        
        - seg: corresponding mask to augment
    
    
    """
    
    def __init__(self, img, seg):
        
        self.img = img
        
        self.seg = seg

    
    
    

    def parameterRandomization(self):
        
        """
        Randomize parameters for scale, translation, rotation and flipping
        
        Params:
            
            - img: image where to apply later transforms
            
            - probs: probabilities for each event
            
            - limit_params: limit parameters for each event
            
            
        Returns: [scale, trans_x, trans_y, angle, flip(1 or 2, 1: horizontal, 2: vertical), time_flip]
        
        """
        
        r_sc = random.uniform(0,1)
        
        r_rot = random.uniform(0,1)
        
        r_flip = random.uniform(0,1)
        
        r_flip_temp = random.uniform(0,1)
        
        
        # Scale randomization
        
        if r_sc < params.augm_probs[0]:
            
            sc = random.uniform(0,params.augm_params[0])
    
            scale = 1 + (np.random.rand()-1) * sc
            
        else:
    
            scale = 1
        
            
        
        # Rotation randomization
        
        if r_rot < params.augm_probs[1]:
            
            angle = random.uniform(- params.augm_params[1], params.augm_params[1])*np.pi/180
    
        else:
            
            angle = 0
        
        
        # Flipping randomization
        
        if r_flip < params.augm_probs[2]:
            
            flip = np.random.randint(1,3)
        
        else:
            
            flip = 0
    
        
        # Temporal flipping randomization
        
        if r_flip_temp < params.augm_probs[3]:
            
            flip_temp = 1
        
        else:
            
            flip_temp = 0
    
    
        out_params = [scale, angle, flip, flip_temp]
        
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


# To run in terminal:

# img = sys.argv[1]
# seg = sys.argv[2] 
# limit_params = sys.argv[3]
# probs = sys.argv[4] 
# studies = sys.argv[5]
# k = sys.argv[6]
# rep = sys.argv[7]    


# if __name__ == "__main__":
        
    