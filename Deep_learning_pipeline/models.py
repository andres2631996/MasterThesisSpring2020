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

from torchvision import models

import numpy as np

from convlstm import ConvLSTM


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

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel + 1, stride = (2,2),\
                padding=(1,1))
            
        self.bn1 = EncoderNorm_2d(out_chan)
            
        #self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel, padding=padding)
        
        
        if params.normalization is not None:
            
            self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        
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

        self.conv1 = nn.ConvTranspose2d(in_chan, out_chan, kernel + 1, stride = (2,2),\
                padding=padding)
        
        self.bn1 = EncoderNorm_2d(out_chan)

        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel, padding=padding)
        
        self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        out = self.bn2(self.conv2(x))

        return F.relu(x + out)
    
    
class CustomPad(nn.Module):
    
    def __init__(self, padding):
        
        super(CustomPad, self).__init__()
    
        self.padding = padding
    
    def forward(self, x):
    
        return F.pad(x, self.padding, mode = 'replicate')
    


    

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
        
        #self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(in_chan, out_chan, 1)
        
        


    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.conv2(x + out)


        return out

    
class addRowCol(nn.Module):
    
    
    def __init__(self):
        
        super(addRowCol, self).__init__()
    
    def forward(self, x):
                
        new_row = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[-1]).cuda()
        
        new_col = torch.zeros(x.shape[0], x.shape[1], x.shape[-1] + 1, 1).cuda()
        
        new = torch.cat((x, new_row), 2)
        
        final = torch.cat((new, new_col),3)
        
        return final


class UNet_with_Residuals(nn.Module):
    
    """
    U-Net with residuals architecture, extracted from Bratt et al., 2019 paper
    
    
    """

    def __init__(self):
        
        super(UNet_with_Residuals, self).__init__()

        self.cat = Concat()
        
        self.pad = addRowCol()
        
        # Decide on number of input channels
        
        if params.sum_work and 'both' in params.train_with:
            
            in_chan = 4 # Train with magnitude + phase + sum of both along time
        
        elif (params.sum_work and not('both' in params.train_with)) or (not(params.sum_work) and 'both' in params.train_with):
            
            in_chan = 2 # Train with magnitude or phase and sum in time // Train with magnitude + phase (no sum)
            
        elif not(params.sum_work) and not('both' in params.train_with):
        
            in_chan = 1 # Train magnitude or phase (no sum)

        self.conv1 = nn.Conv2d(in_chan, params.base, params.kernel_size, padding=params.padding)
                    
        self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        self.bn2 = EncoderNorm_2d(params.base)
                

        self.Rd1 = Res_Down(params.base, params.base*2, params.kernel_size, params.padding)
        self.Rd2 = Res_Down(params.base*2, params.base*2, params.kernel_size, params.padding)
        #self.Rd3 = Res_Down(params.base*4, params.base*4, params.kernel_size, params.padding)
        #self.Rd4 = Res_Down(params.base*8, params.base*8, params.kernel_size, params.padding)
        
        self.fudge = nn.ConvTranspose2d(params.base*4, params.base*4, params.kernel_size, stride = (1,1),\
                padding = params.padding)

        
        #self.Ru3 = Res_Up(params.base*8, params.base*8, params.kernel_size, params.padding)
        #self.Ru2 = Res_Up(params.base*4, params.base*4, params.kernel_size, params.padding)
        self.Ru1 = Res_Up(params.base*2, params.base*2, params.kernel_size, params.padding)
        self.Ru0 = Res_Up(params.base*2, params.base, params.kernel_size, params.padding)

        

#        self.Ru3 = Res_Up(512,512)
#        self.Ru2 = Res_Up(512,256)
#        self.Ru1 = Res_Up(256,128)
#        self.Ru0 = Res_Up(128,64)

        self.Rf = Res_Final(params.base, len(params.class_weights), params.kernel_size, params.padding)

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        e0 = F.relu(self.bn2(self.conv2(out)))
        
        

        e1 = self.Rd1(e0)
        e2 = self.drop(self.Rd2(e1))
        #e3 = self.drop(self.Rd3(e2))
        #e4 = self.Rd4(e3)


        #d3 = self.Ru3(e4)
        d1 = self.Ru1(e2)
        
        if d1.shape[2] != e1.shape[2]:
            
            e1 = self.pad(e1)
        
        #d2 = self.Ru2(self.cat(d3[:,(params.base*4):],e3[:,(params.base*4):]))
        #d1 = self.Ru1(self.cat(d2[:,(params.base*2):],e2[:,(params.base*2):]))
        
        #if d1.shape[2] != e1.shape[2]:
        
            #e1 = self.pad(e1)
            
        d0 = self.Ru0(self.cat(d1[:,params.base:],e1[:,params.base:]))
        
        
        if d0.shape[2] != e0.shape[2]:
        
            e0 = self.pad(e0)

        out = self.Rf(self.cat(e0[:,(params.base//2):],d0[:,(params.base//2):]))


        return out
    
    
    

class pretrainedEncoder(nn.Module):
    
    """
    Pretrained encoder with ResNet18 weights from ImageNet
    
    """
    
    def __init__(self):
        
        super(pretrainedEncoder, self).__init__()
        
        self.resnet = models.resnet18(pretrained=True).cuda()
        
        
    def forward(self,x):
        
        modules = list(self.resnet.children())[:(params.num_layers - 8)]
        
        #resnet = nn.Sequential(*modules)
        
        inter_results = []
        
        layersInterest = [2]
        
        for i in range(len(modules)):
            
            for param in modules[i].parameters():
            
                param.requires_grad = False # Do not need to train this part of the model
            
            x = modules[i](x)
            
            if i in layersInterest:
                
                inter_results.append(x)
        
        return x, inter_results
    

    
class pretrainingPreprocessing(nn.Module):
    
    """
    Preprocesses the data so that it can be properly processed in the pretrained ResNet18
    
    """
    
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        
        super(pretrainingPreprocessing, self).__init__()
    
        self.mean = torch.from_numpy(np.array(mean, dtype=np.float32)).float().to("cuda:0")

        self.std = torch.from_numpy(np.array(std, dtype=np.float32)).float().to("cuda:0")
        
        self.mean = self.mean.unsqueeze(0).unsqueeze(-1) #add a dimenstion for batch and (width*height)
        
        self.std = self.std.unsqueeze(0).unsqueeze(-1)
        
        
        # Decide on number of input channels
        
        if params.sum_work and 'both' in params.train_with:
            
            in_chan = 4 # Train with magnitude + phase + sum of both along time
        
        elif (params.sum_work and not('both' in params.train_with)) or (not(params.sum_work) and 'both' in params.train_with):
            
            in_chan = 2 # Train with magnitude or phase and sum in time // Train with magnitude + phase (no sum)
            
        elif not(params.sum_work) and not('both' in params.train_with):
        
            in_chan = 1 # Train magnitude or phase (no sum)
        
        self.convertconv = nn.Conv2d(in_chan, 3, 1, padding=0) # Turn input without three or one channel into 3 channel-input
    
    def forward(self,x):
        
        # Pretrained encoder requires 3 channels. Modify input so that it has 3 channels

        if x.shape[1] == 1:
            
            x = x.repeat(1, 3, 1, 1) 
        
        else:
            
            if x.shape[1] != 3:
                
                x = self.convertconv(x)
        
        # Tensor normalization
        
        h, w = x.shape[2:]
        
        norm_tensor = x.view(x.shape[0], x.shape[1], -1).cuda() #batch x channel x (height*width)
        
        norm_tensor = norm_tensor - self.mean # Make image mean zero
        
        norm_tensor = norm_tensor / self.std # Make std = 1
        
        norm_tensor = norm_tensor.view(x.shape[0], x.shape[1], h, w) #back to batch x chan x w x h
        
        return norm_tensor
    

    
class UNet_with_ResidualsPretrained(nn.Module):
    
    """
    UNet with skip connections pretrained on ResNet18 (ImageNet) weights in the encoder
    
    """
    
    def __init__(self):
    
        super(UNet_with_ResidualsPretrained, self).__init__()
            
        self.preprocess = pretrainingPreprocessing()

        self.pool = nn.MaxPool2d(2, 2)
        
        self.cat = Concat()
        
        self.bn = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)
        
        self.encoder = pretrainedEncoder()
        
        self.pad = addRowCol()
        
        self.fudge = nn.ConvTranspose2d(params.base, params.base, params.kernel_size, stride = (2,2),\
                padding = params.padding)
        
        #self.Ru2 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
        
        #self.Ru1 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.kernel_size, params.padding)
        
        self.Ru0 = Res_Up(params.base, params.base, params.kernel_size, params.padding)
        
        self.fudge2 = nn.ConvTranspose2d(params.base, params.base, params.kernel_size, stride = (2,2), padding = params.padding)
        
        self.Rf = Res_Final(params.base, len(params.class_weights), params.kernel_size, params.padding)
        
   

    def forward(self,x):
        
        # Preprocess the data so that it can be inputted to the pretrained encoder
        
        x = self.preprocess(x)

        
        # Pretrained encoder
        
        e, inter = self.encoder(x)
        
        #print(e.shape, inter[-1].shape, inter[-2].shape)

        #e = self.fudge(e)
        
        #e = self.pad(e)
        
        #print(e.shape)
        
        # Decoder

        #d2 = self.Ru2(e)
        
        #print(d2.shape, inter[-1].shape)

        #if d2.shape[2] != inter[-1].shape[2]:
        
            #d2 = self.pad(d2)
        
        
        #d1 = self.Ru1(self.cat(e[:,(params.base*2):],inter[-1]))
        
        #if d1.shape[2] != inter[-2].shape[2]:
        
            #d1 = self.pad(d1)
        

        d0 = self.Ru0(e)
        
        # Final layer
        
        if d0.shape[2] != inter[-1].shape[2]:
            
            d0 = self.pad(d0)

        out = self.fudge2(self.cat(inter[-1][:,(params.base//2):],d0[:,(params.base//2):]))
        
        out = self.pad(out)
        
        out = self.Rf(out)


        return out
    
    
class UNetLSTM(nn.Module):
    
    """
    U-Net with convLSTM operators in the encoder, to process 2D+time information
    
    """
    
    
    def __init__(self):
        
        super(UNetLSTM, self).__init__()
        
        self.cat = Concat()
        
        self.pad = addRowCol()
        
        # Reshape
        
        self.conv1 = nn.Conv2d(in_chan, params.base, params.kernel_size, padding=params.padding)
                    
        self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        self.bn2 = EncoderNorm_2d(params.base)
        
        if 'both' in params.train_with:
            
            channels = 2
            
        else:
            
            channels = 1
            
        # Reshape
        
        self.convlstm1 = ConvLSTM(params.base, params.base*2, params.kernel_size, 1, True, True, False)
        
        self.c