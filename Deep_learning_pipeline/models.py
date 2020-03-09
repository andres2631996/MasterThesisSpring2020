#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:47:57 2020

@author: andres
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import params


class Concat(nn.Module):
    
    """
    Tensor concatenation.
    
    Params:
        
        - x1 and x2: tensors to concatenate
    
    Returns:
        
        - cat: concatenated tensors
    
    """
    
    def __init__(self):
        
        super(Concat, self).__init__()

    def forward(self, x1, x2):
        
        cat = torch.cat([x1,x2], dim=1)
        
        return cat
    
    

class EncoderNorm_2d(nn.Module):
    
    
    """
    Normalization for 2D tensors in the encoder section of U-Net
    
    Params:
        
        - x: tensor to normalize
    
    
    """
    
    def __init__(self, channels):
        
        super(EncoderNorm_2d, self).__init__()
        
        if params.normalization == 'instance' or params.normalization == 'Instance' or params.normalization == 'INSTANCE':
        
            self.bn = nn.InstanceNorm2d(channels)
        
        elif params.normalization == 'batch' or params.normalization == 'Batch' or params.normalization == 'BATCH':  
        
            self.bn = nn.BatchNorm2d(channels)
        
        else:
            
            print('Wrong normalization type. Please choose an adequate type (instance/batch)')

    def forward(self, x):
        
        return self.bn(x)
    

class Res_Down(nn.Module):
    
    """
    Compute residuals in encoder convolutional layers.
    
    Params:
        
        - in_chan: number of input feature maps
        
        - out_chan: number of output feature maps
        
        - kernel: kernel size to apply
        
        - padding: padding to apply
    
    
    Returns: down residual convolutional block
    
    
    """
    
    def __init__(self, in_chan, out_chan, kernel, padding):
        
        super(Res_Down, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel, stride = (1,1),\
                padding=(1,1))
            
        self.bn1 = EncoderNorm_2d(out_chan)
            
        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel, padding=padding)
        
        
        if params.normalization is not None:
            
            self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        
        out = self.bn2(self.conv2(x))

        return F.relu(x + out)



class Res_Up(nn.Module):
    
    
    """
    Compute residuals in decoder convolutional layers.
    
    Params:
        
        - in_chan: number of input feature maps
        
        - out_chan: number of output feature maps
        
        - kernel: kernel size to apply
        
        - padding: padding to apply
    
    
    Returns: up residual convolutional block
    
    
    """
    
    def __init__(self, in_chan, out_chan, kernel, padding):
        
        super(Res_Up, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_chan, out_chan, kernel, stride = (1,1),\
                padding=padding)
        
        self.bn1 = EncoderNorm_2d(out_chan)
        
        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=padding)
        
        self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        
        out = self.bn2(self.conv2(x))

        return F.relu(x + out)
    

class Res_Final(nn.Module):
    
    """
    
    Compute residuals in final convolutional layers.
    
    Params:
        
        - in_chan: number of input feature maps
        
        - out_chan: number of output feature maps
        
        - kernel: kernel size to apply
        
        - padding: padding to apply
    
    
    Returns: up residual convolutional block
    
    
    """
    
    def __init__(self, in_chan, out_chan, kernel, padding):
        
        super(Res_Final, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, in_chan, kernel, padding=padding)
        
        self.bn1 = EncoderNorm_2d(in_chan)
        
        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(in_chan, out_chan, 1)
        
        


    def forward(self, x):
        
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        
        out = self.conv2(x + out)


        return out


class UNet_with_Residuals(nn.Module):
    
    """
    U-Net with residuals architecture, extracted from Bratt et al., 2019 paper
    
    
    """

    def __init__(self, outs):
        
        super(UNet_with_Residuals, self).__init__()

        self.cat = Concat()
        
        if 'both' in params.train_with: # There are two channels

            self.conv1 = nn.Conv2d(4, params.base, params.kernel_size, padding=params.padding)
        
        else:
            
            self.conv1 = nn.Conv2d(2, params.base, params.kernel_size, padding=params.padding)
            
        self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        self.bn2 = EncoderNorm_2d(params.base)
                

        self.Rd1 = Res_Down(params.base, params.base*2, params.kernel_size, params.padding)
        self.Rd2 = Res_Down(params.base*2, params.base*4, params.kernel_size, params.padding)
        self.Rd3 = Res_Down(params.base*4, params.base*4, params.kernel_size, params.padding)
        #self.Rd4 = Res_Down(params.base*8, params.base*8, params.kernel_size, params.padding)
        
        self.fudge = nn.ConvTranspose2d(params.base*4, params.base*4, params.kernel_size, stride = (1,1),\
                padding = params.padding)

        
        #self.Ru3 = Res_Up(params.base*8, params.base*8, params.kernel_size, params.padding)
        self.Ru2 = Res_Up(params.base*4, params.base*4, params.kernel_size, params.padding)
        self.Ru1 = Res_Up(params.base*4, params.base*2, params.kernel_size, params.padding)
        self.Ru0 = Res_Up(params.base*2, params.base, params.kernel_size, params.padding)

        

#        self.Ru3 = Res_Up(512,512)
#        self.Ru2 = Res_Up(512,256)
#        self.Ru1 = Res_Up(256,128)
#        self.Ru0 = Res_Up(128,64)

        self.Rf = Res_Final(params.base, outs, params.kernel_size, params.padding)

    def forward(self, x):
        
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        
        e0 = F.relu(self.bn2(self.conv2(out)))
        


        e1 = self.Rd1(e0)
        e2 = self.Rd2(e1)
        e3 = self.Rd3(e2)
        #e4 = self.Rd4(e3)
            
        

        #d3 = self.Ru3(e4)
        d2 = self.Ru2(e3)
        #d2 = self.Ru2(self.cat(d3[:,(params.base*4):],e3[:,(params.base*4):]))
        d1 = self.Ru1(self.cat(d2[:,(params.base*2):],e2[:,(params.base*2):]))
        d0 = self.Ru0(self.cat(d1[:,params.base:],e1[:,params.base:]))

        out = self.Rf(self.cat(e0[:,(params.base//2):],d0[:,(params.base//2):]))


        return out