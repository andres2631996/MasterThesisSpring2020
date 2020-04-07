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

from convRNNs import ConvLSTM, ConvGRU

import cc3d

from skimage import measure

import utilities


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
        
        if params.architecture == 'NewUNet_with_Residuals':
            
            self.conv1 = nn.Conv2d(in_chan, out_chan, kernel, stride = (1,1),\
                    padding=padding)
        
        else:

            self.conv1 = nn.Conv2d(in_chan, out_chan, kernel + 1, stride = (2,2),\
                    padding=padding)
        
        if params.normalization is not None:
            
            self.bn1 = EncoderNorm_2d(out_chan)
            
        #self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel, padding=padding)
        
        
        if params.normalization is not None:
            
            self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        if params.normalization is not None:
        
            x = F.relu(self.bn1(self.conv1(x)))

            out = self.bn2(self.conv2(x))
            
        else:
            
            x = F.relu(self.conv1(x))
            
            out = self.conv2(x)

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
        
        if params.architecture == 'NewUNet_with_Residuals':
            
            self.conv1 = nn.ConvTranspose2d(in_chan, out_chan, kernel, stride = (1,1),\
                    padding=padding)
            
        else:

            self.conv1 = nn.ConvTranspose2d(in_chan, out_chan, kernel + 1, stride = (2,2),\
                    padding=padding)
        
        if params.normalization is not None:
        
            self.bn1 = EncoderNorm_2d(out_chan)

        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel, padding=padding)
        
        if params.normalization is not None:
        
            self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        if params.normalization is not None:
        
            x = F.relu(self.bn1(self.conv1(x)))

            out = self.bn2(self.conv2(x))
            
        else:
            
            x = F.relu(self.conv1(x))

            out = self.conv2(x)

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
        
        if params.normalization is not None:
        
            self.bn1 = EncoderNorm_2d(in_chan)
        
        #self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(in_chan, out_chan, 1)
        
        


    def forward(self, x):
        
        if params.normalization is not None:
        
            out = F.relu(self.bn1(self.conv1(x)))
        
        else:
            
            out = F.relu(self.conv1(x))
        
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
    
    
    
    
class distanceLayer(nn.Module):
    
    """
    Distance processing layer placed at the end of the architecture encoder
    
    Params:
    
        - x: tensor to be processed in the distance processing layer
    
    """
    
    
    def __init__(self, ch_in, ch_out):
        
        super(distanceLayer, self).__init__()
        
        self.cat = Concat()
        
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True)
        
        self.bn = EncoderNorm_2d(ch_out)
        
    
    def forward(self,x):
        
        distance_maps = utilities.distanceTransform(x, 'net')
        
        distance_maps = torch.tanh(distance_maps)
        
        concat = self.cat(x,distance_maps.float())
        
        conv_out = self.bn(self.conv(concat))
        
        return conv_out
    


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
            
            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time
        
        elif (params.sum_work and not('both' in params.train_with)):
            
            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time
            
        elif (not(params.sum_work) and 'both' in params.train_with):
            
            in_chan = 2 # Train with magnitude + phase (no sum)
            
        elif not(params.sum_work) and not('both' in params.train_with):
        
            in_chan = 1 # Train magnitude or phase (no sum)

        self.conv1 = nn.Conv2d(in_chan, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
                    
            self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
        
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
        
        if params.normalization is not None:
        
            out = F.relu(self.bn1(self.conv1(x)))

            e0 = F.relu(self.bn2(self.conv2(out)))
            
        else:
        
            out = F.relu(self.conv1(x))

            e0 = F.relu(self.conv2(out))

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
    
    
class UNet_with_ResidualsFourLayers(nn.Module):
    
    """
    U-Net with residuals architecture, extracted from Bratt et al., 2019 paper, with four layers
    
    
    """

    def __init__(self):
        
        super(UNet_with_ResidualsFourLayers, self).__init__()

        self.cat = Concat()
        
        self.pad = addRowCol()
        
        # Decide on number of input channels
        
        if params.sum_work and 'both' in params.train_with:
            
            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time
        
        elif (params.sum_work and not('both' in params.train_with)):
            
            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time
            
        elif (not(params.sum_work) and 'both' in params.train_with):
            
            in_chan = 2 # Train with magnitude + phase (no sum)
            
        elif not(params.sum_work) and not('both' in params.train_with):
        
            in_chan = 1 # Train magnitude or phase (no sum)

        self.conv1 = nn.Conv2d(in_chan, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
                    
            self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
        
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

        self.Rf = Res_Final(params.base, len(params.class_weights), params.kernel_size, params.padding)


    def forward(self, x):
        
        if params.normalization is not None:
        
            out = F.relu(self.bn1(self.conv1(x)))

            e0 = F.relu(self.bn2(self.conv2(out)))
            
        else:
        
            out = F.relu(self.conv1(x))

            e0 = F.relu(self.conv2(out))

        e1 = self.Rd1(e0)
        e2 = self.drop(self.Rd2(e1))
        e3 = self.drop(self.Rd3(e2))
        #e4 = self.Rd4(e3)


        #d3 = self.Ru3(e4)
        d2 = self.Ru2(e3)
        
        if d2.shape[2] != e2.shape[2]:
            
            e2 = self.pad(e2)
        
        #d2 = self.Ru2(self.cat(d3[:,(params.base*4):],e3[:,(params.base*4):]))
        d1 = self.Ru1(self.cat(d2[:,(params.base*2):],e2[:,(params.base*2):]))
        
        if d1.shape[2] != e1.shape[2]:
        
            e1 = self.pad(e1)
            
        d0 = self.Ru0(self.cat(d1[:,params.base:],e1[:,params.base:]))
        
        
        if d0.shape[2] != e0.shape[2]:
        
            e0 = self.pad(e0)

        out = self.Rf(self.cat(e0[:,(params.base//2):],d0[:,(params.base//2):]))



        return out
    
    
class connectedComponents(nn.Module):
    
    """
    Extract the number of connected components of the result from a neural network.
    
    Take the connected component with the largest probability to be renal artery and
    
    leave just that connected component.
    
    Params:
    
        - x: tensor result from neural network (B, C, H, W) or (B, C, H, W, T)
        
    Returns:
    
        - out: tensor result with just one connected component
    
    """ 
    
    def __init__(self):
        
        super(connectedComponents, self).__init__()
        
    def forward(self,x):
        
        # Compute output tensor from probability classes
        
        binary = torch.argmax(x, 1)
        
        # Transform tensor into Numpy array
        
        binary_array = binary.detach().numpy() # B,H,W (T)
        
        x_array = x.detach().numpy() # B,C,H,W (T)
        
        # Get labels from connected components for all elements in batch
        
        if len(binary_array.shape) == 3: # 2D arrays
            
            for i in range(binary_array.shape[0]):
            
                labels = measure.label(binary_array[i,:,:], background=0)
        
        elif len(x_array.shape) == 4: # 3D arrays

            for i in range(binary_array.shape[0]):
            
                labels = cc3d.connected_components(binary_array[i,:,:]) # 26-connected
                
                
                
    
class NewUNet_with_Residuals(nn.Module):
    
    """
    U-Net with residuals architecture, extracted from Bratt et al., 2019 paper
    
    
    """

    def __init__(self):
        
        super(NewUNet_with_Residuals, self).__init__()

        self.cat = Concat()
        
        self.pad = addRowCol()
        
        # Decide on number of input channels
        
        if params.sum_work and 'both' in params.train_with:
            
            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time
        
        elif (params.sum_work and not('both' in params.train_with)):
            
            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time
            
        elif (not(params.sum_work) and 'both' in params.train_with):
            
            in_chan = 2 # Train with magnitude + phase (no sum)
            
        elif not(params.sum_work) and not('both' in params.train_with):
        
            in_chan = 1 # Train magnitude or phase (no sum)

        self.conv1 = nn.Conv2d(in_chan, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
                    
            self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
        
            self.bn2 = EncoderNorm_2d(params.base)
                

        self.Rd1 = Res_Down(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.kernel_size, params.padding)
        self.Rd2 = Res_Down(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
        self.Rd3 = Res_Down(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
        #self.Rd4 = Res_Down(params.base*8, params.base*8, params.kernel_size, params.padding)
        
        self.fudge = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = (1,1),\
                padding = params.padding)

        
        #self.Ru3 = Res_Up(params.base*8, params.base*8, params.kernel_size, params.padding)
        self.Ru2 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
        self.Ru1 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.kernel_size, params.padding)
        self.Ru0 = Res_Up(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.kernel_size, params.padding)

        

#        self.Ru3 = Res_Up(512,512)
#        self.Ru2 = Res_Up(512,256)
#        self.Ru1 = Res_Up(256,128)
#        self.Ru0 = Res_Up(128,64)

        self.Rf = Res_Final(params.base, len(params.class_weights), params.kernel_size, params.padding)


    def forward(self, x):
        
        if params.normalization is not None:
        
            out = F.relu(self.bn1(self.conv1(x)))

            e0 = F.relu(self.bn2(self.conv2(out)))
            
        else:
        
            out = F.relu(self.conv1(x))

            e0 = F.relu(self.conv2(out))

        e1 = self.Rd1(e0)
        e2 = self.drop(self.Rd2(e1))
        e3 = self.drop(self.Rd3(e2))
        #e4 = self.Rd4(e3)


        #d3 = self.Ru3(e4)
        d2 = self.Ru2(e3)
        
        #d2 = self.Ru2(self.cat(d3[:,(params.base*4):],e3[:,(params.base*4):]))
        d1 = self.Ru1(self.cat(d2[:,(params.base*(2**(params.num_layers - 2))):],e2[:,(params.base*(2**(params.num_layers - 2))):]))

            
        d0 = self.Ru0(self.cat(d1[:,params.base*(2**(params.num_layers - 3)):],e1[:,params.base*(2**(params.num_layers - 3)):]))

        out = self.Rf(self.cat(e0[:,params.base//2:],d0[:,params.base//2:]))


        return out 
    
    
    
class conv_res_block(nn.Module):
    
    """
    Convolutional block with residuals for Attention U-Net.
    
    Can optionally be used with recurrent units, too
    
    Params:
    
    - Init: ch_in: number of input channels // ch_out: number of output channels // key: flag stating if the block is in the encoder or in the decoder
    
    - Forward: x: input tensor
    
    Outputs: x: output tensor
    
    
    """
    
    def __init__(self,ch_in,ch_out, key):
        
        super(conv_res_block,self).__init__()
        
        if params.rnn == None:
        
            self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True)
            
        else:
            
            if params.rnn_position == 'full' or params.rnn_position == key:
            
                if params.rnn == 'LSTM':

                    self.conv1 = ConvLSTM(ch_in, ch_out, (params.kernel_size, params.kernel_size), 1, True, True, False)

                elif params.rnn == 'GRU':

                    if torch.cuda.is_available():

                        dtype = torch.cuda.FloatTensor # computation in GPU

                    else:

                        dtype = torch.FloatTensor

                    self.conv1 = ConvGRU(ch_in, ch_out, (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
              
            else:
                
                self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=params.kernel_size, stride=1, padding=params.padding, bias=True)
        
        self.bn1 = EncoderNorm_2d(ch_out)
        
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True)
        
        self.bn2 = EncoderNorm_2d(ch_out)

        self.relu2 = nn.ReLU(inplace=True)
        
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=params.kernel_size,stride=1,padding=1)


    def forward(self,x):
        
        x0 = self.Conv_1x1(x)
        
        if params.rnn is not None:
            
            x = x.view(params.batch_size, x.shape[0]//params.batch_size, x.shape[-3], x.shape[-2], x.shape[-1])
        
            x,_ = list(self.conv1(x))
        
            x = x[0].view(x[0].shape[0]*x[0].shape[1],x[0].shape[-3], x[0].shape[-2], x[0].shape[-1])
            
            x1 = self.relu1(self.bn1(x))
            
        else:
        
            x1 = self.relu1(self.bn1(self.conv1(x)))
        
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        
        return x0 + x2
    
    
class up_conv(nn.Module):
    
    """
    Perform upsampling in decoder section of U-Net
    
    Params:
    
        - init: ch_in: input number of channels // ch_out: output number of channels
    
        - forward: x: input tensor
        
    """
    
    
    def __init__(self,ch_in,ch_out):
        
        super(up_conv,self).__init__()
        
        self.up = nn.ConvTranspose2d(ch_in, ch_in, params.kernel_size, stride = 2,padding=params.padding)
        
        self.bn1 = EncoderNorm_2d(ch_in)

        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True)
        
        self.bn2 = EncoderNorm_2d(ch_out)

        self.relu2 = nn.ReLU(inplace=True)
        

    def forward(self,x):
        
        x1 = self.relu1(self.bn1(self.up(x)))
        
        x2 = self.relu2(self.bn2(self.conv(x1)))

        
        return x2
    
    
       
    
class Attention_block(nn.Module):
    
    def __init__(self,F_g,F_l,F_int):
        
        super(Attention_block,self).__init__()
        
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        
        self.bn1 = EncoderNorm_2d(F_int)
            
        
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True)
            
        self.bn2 = EncoderNorm_2d(F_int)
        
        

        self.psi = nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True)
        
        self.bn3 = EncoderNorm_2d(1)
        
        self.sigma = nn.Sigmoid()
        
        self.relu = nn.ReLU(inplace=True)
        
        
        
    def forward(self,g,x):
        
        g1 = self.bn1(self.W_g(g))
        
        x1 = self.bn2(self.W_x(x))
        
        psi = self.relu(g1+x1)
        
        psi = self.sigma(self.bn3(self.psi(psi)))

        return x*psi 
    
    
    

class AttentionUNet(nn.Module):
    
    """
    U-Net with residuals architecture, extracted from Bratt et al., 2019 paper, including Attention Gates, from Oktay et al., 2018 paper
    
    Processes 2D data
    
    """

    def __init__(self):
        
        super(AttentionUNet, self).__init__()

        self.cat = Concat()
        
        self.pad = addRowCol()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # Decide on number of input channels
        
        if params.three_D or (not(params.three_D) and params.add3d > 0):
        
            if 'both' in params.train_with:

                in_chan = 2

            else:

                in_chan = 1
                
        else:
        
            if params.sum_work and 'both' in params.train_with:

                in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time

            elif (params.sum_work and not('both' in params.train_with)):

                in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time

            elif (not(params.sum_work) and 'both' in params.train_with):

                in_chan = 2 # Train with magnitude + phase (no sum)

            elif not(params.sum_work) and not('both' in params.train_with):

                in_chan = 1 # Train magnitude or phase (no sum)
            
        # Encoder

        self.Down1 = conv_res_block(ch_in=in_chan, ch_out=params.base*(2**(params.num_layers - 3)), key = 'encoder')
        
        self.Down2 = conv_res_block(ch_in=params.base*(2**(params.num_layers - 3)),ch_out=params.base*(2**(params.num_layers - 2)), key = 'encoder')
    
        self.Down3 = conv_res_block(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 1)), key = 'encoder')
        
        self.drop = nn.Dropout2d(params.dropout)
   
        
        # Decoder + Attention Gates                                           
                                                   
        self.Up3 = up_conv(ch_in=params.base*(2**(params.num_layers - 1)),ch_out=params.base*(2**(params.num_layers - 2)))
                                                   
        self.Att3 = Attention_block(F_g=params.base*(2**(params.num_layers - 2)),F_l=params.base*(2**(params.num_layers - 2)),F_int=params.base*(2**(params.num_layers - 3)))
                                                   
        self.Up_conv3 = conv_res_block(ch_in=params.base*(2**(params.num_layers - 1)), ch_out=params.base*(2**(params.num_layers - 2)), key = 'decoder')
        
        self.Up2 = up_conv(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 3)))
            
        self.Att2 = Attention_block(F_g=params.base*(2**(params.num_layers - 3)),F_l=params.base*(2**(params.num_layers - 3)),F_int=int(params.base*(2**(params.num_layers - 4))))
                                                   
        self.Up_conv2 = conv_res_block(ch_in=params.base*(2**(params.num_layers - 2)), ch_out=params.base*(2**(params.num_layers - 3)), key = 'decoder')

        self.Conv_1x1 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1,  stride=1,padding=0)
                           
                           
    def forward(self,x):
                           
        if params.three_D or (not(params.three_D) and params.add3d > 0):
                           
            # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)

            x = x.view(x.shape[0]*x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
                           
        # encoding path
                           
        x1 = self.Down1(x)

        x2 = self.Maxpool(x1)
        x2 = self.drop(self.Down2(x2))
        
        x3 = self.Maxpool(x2)
        x3 = self.drop(self.Down3(x3))
        
        
        d3 = self.Up3(x3)
        
        if d3.shape[2] != x2.shape[2]:
            
            d3 = self.pad(d3)
        
        x2 = self.Att3(g=d3,x=x2)
        d3 = self.Up_conv3(self.cat(x2,d3))

        d2 = self.Up2(d3)
        
        if d2.shape[2] != x1.shape[2]:
            
            d2 = self.pad(d2)
        
        x1 = self.Att2(g=d2,x=x1)
        d2 = self.Up_conv2(self.cat(x1,d2))

        d1 = self.Conv_1x1(d2)
        
        if params.three_D or (not(params.three_D) and params.add3d > 0):
            
            d1 = d1.view(params.batch_size, d1.shape[1], d1.shape[2], d1.shape[3], d1.shape[0]//params.batch_size)

        return d1 
                           
                           
class NewAttentionUNet(nn.Module):
    
    """
    U-Net with residuals architecture, extracted from Bratt et al., 2019 paper, including Attention Gates, from Oktay et al., 2018 paper. Does not include subsamplings nor upsamplings
    
    Processes 2D data
    
    """

    def __init__(self):
        
        super(NewAttentionUNet, self).__init__()

        self.cat = Concat()
        
        self.pad = addRowCol()
        
        # Decide on number of input channels
        
        if params.three_D or (not(params.three_D) and params.add3d > 0):
        
            if 'both' in params.train_with:

                in_chan = 2

            else:

                in_chan = 1
                
        else:
        
            if params.sum_work and 'both' in params.train_with:

                in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time

            elif (params.sum_work and not('both' in params.train_with)):

                in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time

            elif (not(params.sum_work) and 'both' in params.train_with):

                in_chan = 2 # Train with magnitude + phase (no sum)

            elif not(params.sum_work) and not('both' in params.train_with):

                in_chan = 1 # Train magnitude or phase (no sum)
            
        # Encoder

        self.Down1 = conv_res_block(ch_in=in_chan, ch_out=params.base*(2**(params.num_layers - 3)), key = 'encoder')
        
        self.Down2 = conv_res_block(ch_in=params.base*(2**(params.num_layers - 3)),ch_out=params.base*(2**(params.num_layers - 2)), key = 'encoder')
    
        self.Down3 = conv_res_block(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 1)), key = 'encoder')
       
    
        self.drop = nn.Dropout2d(params.dropout)
        
        # Decoder + Attention Gates                                           
                                                   
        self.Att3 = Attention_block(F_g=params.base*(2**(params.num_layers - 2)),F_l=params.base*(2**(params.num_layers - 2)),F_int=params.base*(2**(params.num_layers - 3)))
                                                   
        self.Up_conv3 = conv_res_block(ch_in=params.base*(2**(params.num_layers - 1)), ch_out=params.base*(2**(params.num_layers - 2)), key = 'decoder')
            
        self.Att2 = Attention_block(F_g=params.base*(2**(params.num_layers - 3)),F_l=params.base*(2**(params.num_layers - 3)),F_int=int(params.base*(2**(params.num_layers - 4))))
                                                   
        self.Up_conv2 = conv_res_block(ch_in=params.base*(2**(params.num_layers - 2)), ch_out=params.base*(2**(params.num_layers - 3)), key = 'decoder')

        self.Conv_1x1 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
                           
    def forward(self,x):
                           
        if params.three_D or (not(params.three_D) and params.add3d > 0):
                           
            # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)

            x = x.view(x.shape[0]*x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
                           
        # encoding path
                           
        x1 = self.Down1(x)

        x2 = self.drop(self.Down2(x1))
        
        x3 = self.drop(self.Down3(x2))
        
        d3 = self.Up_conv3(x3)
        
        x2 = self.Att3(g=d3,x=x2)
        
        d2 = self.Up_conv2(self.cat(x2[:,(params.base*(2**(params.num_layers - 3))):],d3[:,(params.base*(2**(params.num_layers - 3))):]))
        
        x1 = self.Att2(g=d2,x=x1)

        d1 = self.Conv_1x1(self.cat(x1[:,(int(params.base*(2**(params.num_layers - 4)))):],d2[:,(int(params.base*(2**(params.num_layers - 4)))):]))
        
        if params.three_D or (not(params.three_D) and params.add3d > 0):
            
            d1 = d1.view(params.batch_size, d1.shape[1], d1.shape[2], d1.shape[3], d1.shape[0]//params.batch_size)
            

        return d1                            


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
            
            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time
        
        elif (params.sum_work and not('both' in params.train_with)):
            
            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time
            
        elif (not(params.sum_work) and 'both' in params.train_with):
            
            in_chan = 2 # Train with magnitude + phase (no sum)
            
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
        
        if params.normalization is not None:
        
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
    

    
    
class UNetRNNDown(nn.Module):
    
    """
    Encoder layers of U-Net with convLSTMs
    
    """
    
    def __init__(self, in_chan, out_chan):
        
        super(UNetRNNDown, self).__init__()
        
        if params.rnn == 'LSTM':

            self.conv1 = ConvLSTM(in_chan, out_chan, (params.kernel_size, params.kernel_size), 1, True, True, False)
            
        elif params.rnn == 'GRU':
            
            if torch.cuda.is_available():
                
                dtype = torch.cuda.FloatTensor # computation in GPU
                
            else:
                
                dtype = torch.FloatTensor
            
            self.conv1 = ConvGRU(in_chan, out_chan, (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
        else:
             
            print('Wrong recurrent cell introduced. Please introduce a valid name for recurrent cell')
            
            
        
        if params.normalization is not None:
            
            self.bn1 = EncoderNorm_2d(out_chan)
            
        #self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(out_chan, out_chan, params.kernel_size + 1, stride = (2,2), padding = params.padding)
        
        if params.normalization is not None:
            
            self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        # Reshape x to enter the convLSTM cell: (BxT,C,H,W) --> (B,T,C,H,W)
        
        x = x.view(params.batch_size, x.shape[0]//params.batch_size, x.shape[-3], x.shape[-2], x.shape[-1])
        
        # Introduce x into convLSTM
        
        x,_ = list(self.conv1(x))
        
        # Reshape x again to original shape
        
        x = x[0].view(x[0].shape[0]*x[0].shape[1],x[0].shape[-3], x[0].shape[-2], x[0].shape[-1])
        
        if params.normalization is not None:
        
            x = self.bn2(self.conv2(F.leaky_relu(self.bn1(x))))
            
        else:
            
            x = self.conv2(F.leaky_relu(x))

        return F.leaky_relu(x)
    
    
class UNetRNNUp(nn.Module):
    
    """
    Decoder layers of U-Net with convLSTMs or convGRUs
    
    """
    
    def __init__(self, in_chan, out_chan):
        
        super(UNetRNNUp, self).__init__()
        
        if params.rnn == 'LSTM':

            self.conv1 = ConvLSTM(in_chan, out_chan, (params.kernel_size, params.kernel_size), 1, True, True, False)
            
        elif params.rnn == 'GRU':
            
            if torch.cuda.is_available():
                
                dtype = torch.cuda.FloatTensor # computation in GPU
                
            else:
                
                dtype = torch.FloatTensor
            
            self.conv1 = ConvGRU(in_chan, out_chan, (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
        else:
             
            print('Wrong recurrent cell introduced. Please introduce a valid name for recurrent cell')
        
        if params.normalization is not None:
            
            self.bn1 = EncoderNorm_2d(out_chan)
            
        #self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.ConvTranspose2d(out_chan, out_chan, params.kernel_size + 1, stride = (2,2), padding=(1,1))
        
        if params.normalization is not None:
            
            self.bn2 = EncoderNorm_2d(out_chan)


    def forward(self, x):
        
        # Reshape x to enter the convLSTM cell: (BxT,C,H,W) --> (B,T,C,H,W)
        
        x = x.view(params.batch_size, x.shape[0]//params.batch_size, x.shape[-3], x.shape[-2], x.shape[-1])
        
        # Introduce x into convLSTM
        
        x,_ = list(self.conv1(x))
        
        # Reshape x again to original shape
        
        x = x[0].view(x[0].shape[0]*x[0].shape[1],x[0].shape[-3], x[0].shape[-2], x[0].shape[-1])
        
        if params.normalization is not None:
        
            x = self.bn2(self.conv2(F.leaky_relu(self.bn1(x))))

            
        else:
            
            x = self.conv2(F.leaky_relu(x))

        return F.leaky_relu(x)
    
    
    
class UNetRNN(nn.Module):
    
    """
    U-Net with convLSTM or convGRU operators, to process 2D+time information
    
    """
    
    
    def __init__(self):
        
        super(UNetRNN, self).__init__()
        
        self.cat = Concat()
        
        self.pad = addRowCol()
        
        if 'both' in params.train_with:
            
            channels = 2
            
        else:
            
            channels = 1
        
        self.conv1 = nn.Conv2d(channels, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
                    
            self.bn1 = EncoderNorm_2d(params.base)

        self.drop = nn.Dropout2d(params.dropout)

        self.conv2 = nn.Conv2d(params.base, params.base, params.kernel_size, padding=params.padding)
        
        if params.normalization is not None:
        
            self.bn2 = EncoderNorm_2d(params.base)
            
        if params.rnn_position == 'encoder' or params.rnn_position == 'full': 
            
            self.Rd1 = UNetRNNDown(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)))

            self.Rd2 = UNetRNNDown(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)))

            self.Rd3 = UNetRNNDown(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)))
            
        else:
            
            self.Rd1 = Res_Down(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.kernel_size, params.padding)
            self.Rd2 = Res_Down(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
            self.Rd3 = Res_Down(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)
            
        
        
        if params.rnn_position == 'decoder' or params.rnn_position == 'full':
            
            self.Ru3 = UNetRNNUp(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)))

            self.Ru2 = UNetRNNUp(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)))

            self.Ru1 = UNetRNNUp(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)))
            
        else:
                
            self.Ru3 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, params.padding)

            self.Ru2 = Res_Up(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.kernel_size, params.padding)

            self.Ru1 = Res_Up(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.kernel_size, params.padding)
                
        self.Rf = Res_Final(params.base, len(params.class_weights), params.kernel_size, params.padding)
        
        
    def forward(self, x):
        
        # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)
        
        x = x.view(x.shape[0]*x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
        
        # First convolutional layer
        
        if params.normalization is not None:
        
            out = F.relu(self.bn1(self.conv1(x)))

            e0 = F.relu(self.bn2(self.conv2(out)))
            
        else:
        
            out = F.relu(self.conv1(x))

            e0 = F.relu(self.conv2(out))
            
        # Encoder layers: convRNN + conv2D

        e1 = self.Rd1(e0)
        
        e2 = self.drop(self.Rd2(e1))
        
        e3 = self.drop(self.Rd3(e2))
    
        # Decoder layers: conv2Dtranspose + conv2d
        
        d3 = self.Ru3(e3)

        if d3.shape[2] != e2.shape[2]:
                    
            e2 = self.pad(e2)
                
        d2 = self.Ru2(self.cat(d3[:,(params.base*(2**(params.num_layers - 2))):],e2[:,(params.base*(2**(params.num_layers - 2))):]))
        
        if d2.shape[2] != e1.shape[2]:
            
            e1 = self.pad(e1)
            
        d1 = self.Ru1(self.cat(d2[:,(params.base*(2**(params.num_layers - 3))):],e1[:,(params.base*(2**(params.num_layers - 3))):]))
                
        # Final output layer
        
        if d1.shape[2] != e0.shape[2]:
        
            e0 = self.pad(e0)

        out = self.Rf(self.cat(e0[:,(params.base//2):],d1[:,(params.base//2):]))
        
        # Reshape output to original dimensions for loss function computation with respect to corresponding mask
        
        out = out.view(params.batch_size, out.shape[1], out.shape[2], out.shape[3], out.shape[0]//params.batch_size)

        
        return out
    
