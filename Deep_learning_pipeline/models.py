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

import matplotlib.pyplot as plt


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
    

class ConcatTime(nn.Module):
    
    """
    Tensor concatenation.
    
    Params:
        
        - x1 and x2: tensors to concatenate
    
    Returns:
        
        - cat: concatenated tensors
    
    """
    
    def __init__(self):
        
        super(ConcatTime, self).__init__()

    def forward(self, x1, x2):
        
        cat = torch.cat([x1,x2], dim=2)
        
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
    
    
class TimeDistributed(nn.Module):
    
    """
    Transform any regular PyTorch layer into a time-distributed layer
    
    """
    
    def __init__(self, module):
        
        super(TimeDistributed, self).__init__()
        
        self.module = module


    def forward(self, x):

        if len(x.size()) <= 2:
            
            return self.module(x)
        
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(x.size(0)*x.size(1), x.size(2), x.size(-2), x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        y = y.contiguous().view(x.size(0), x.size(1), y.size(1), y.size(-2), y.size(-1))  # (samples, timesteps, output_size)

        
        
        return y
    
    


    

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
    
    
class addRowCol2d(nn.Module):
    
    def __init__(self):
        
        super(addRowCol2d, self).__init__()
    
    def forward(self, x, ref_shape):
        
        final = torch.zeros(ref_shape).cuda()
        
        final = F.interpolate(x, size=(ref_shape[-2], ref_shape[-1]), mode = 'bilinear', align_corners = False)
        
        return final
    
    
    
class addRowCol3d(nn.Module):
    
    
    def __init__(self):
        
        super(addRowCol3d, self).__init__()
    
    def forward(self, x, ref_shape):
        
        final = torch.zeros(ref_shape).cuda()
        
        final = F.interpolate(x, size=(ref_shape[-3], ref_shape[-2], ref_shape[-1]), mode = 'trilinear', align_corners = False)
        
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
                padding = params.padding, output_padding = 1)

        
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
        
        #if d1.shape[2] != e1.shape[2]:
            
         #   e1 = self.pad(e1)
        
        #d2 = self.Ru2(self.cat(d3[:,(params.base*4):],e3[:,(params.base*4):]))
        #d1 = self.Ru1(self.cat(d2[:,(params.base*2):],e2[:,(params.base*2):]))
        
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
        
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=params.kernel_size,stride=1,padding=params.padding)


    def forward(self,x, key):
        
        x0 = self.Conv_1x1(x)
        
        if params.rnn_position == 'full' or params.rnn_position == key:
            
            x = x.view(params.batch_size, x.shape[0]//params.batch_size, x.shape[-3], x.shape[-2], x.shape[-1])
        
            x,_ = list(self.conv1(x))
        
            x = x[0].view(x[0].shape[0]*x[0].shape[1],x[0].shape[-3], x[0].shape[-2], x[0].shape[-1])
            
            x1 = self.relu1(self.bn1(x))
            
        else:
        
            x1 = self.relu1(self.bn1(self.conv1(x)))
        
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        
        return x0 + x2
    
    
    
    
    
class conv_res_blockTD(nn.Module):
    
    """
    Convolutional block with residuals for Attention U-Net. Include time-distributed layers for time data processing
    
    Can optionally be used with recurrent units, too
    
    Params:
    
    - Init: ch_in: number of input channels // ch_out: number of output channels // key: flag stating if the block is in the encoder or in the decoder
    
    - Forward: x: input tensor
    
    Outputs: x: output tensor
    
    
    """
    
    def __init__(self,ch_in,ch_out, key):
        
        super(conv_res_blockTD,self).__init__()
            
        if params.rnn_position == 'full' or params.rnn_position == key:

            if params.rnn == 'LSTM':

                self.conv1 = ConvLSTM(ch_in, ch_out, (params.kernel_size , params.kernel_size), 1, True, True, False)

            elif params.rnn == 'GRU':

                if torch.cuda.is_available():

                    dtype = torch.cuda.FloatTensor # computation in GPU

                else:

                    dtype = torch.FloatTensor

                self.conv1 = ConvGRU(ch_in, ch_out, (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)

        else:

            self.conv1 = TimeDistributed(nn.Conv2d(ch_in, ch_out, kernel_size=params.kernel_size, stride=1, padding=params.padding, bias=True))
        
        self.bn1 = TimeDistributed(nn.InstanceNorm2d(ch_out))
        
        self.relu1 = TimeDistributed(nn.LeakyReLU(inplace=True))
        
        self.conv2 = TimeDistributed(nn.Conv2d(ch_out, ch_out, kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True))
        
        self.bn2 = TimeDistributed(nn.InstanceNorm2d(ch_out))

        self.relu2 = TimeDistributed(nn.LeakyReLU(inplace=True))
        
        self.Conv_1x1 = TimeDistributed(nn.Conv2d(ch_in,ch_out,kernel_size=params.kernel_size,stride=1,padding=params.padding))


    def forward(self,x, key):
        
        x0 = self.Conv_1x1(x)
        
        if params.rnn_position == 'full' or params.rnn_position == key:
        
            x,_ = list(self.conv1(x))
            
            x1 = self.relu1(self.bn1(x[0]))
            
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
        
        self.up = nn.ConvTranspose2d(ch_in, ch_in, params.kernel_size, stride = 2,padding=params.padding, output_padding = 1)
        
        self.bn1 = EncoderNorm_2d(ch_in)

        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True)
        
        self.bn2 = EncoderNorm_2d(ch_out)

        self.relu2 = nn.ReLU(inplace=True)
        

    def forward(self,x):
        
        x1 = self.relu1(self.bn1(self.up(x)))
        
        x2 = self.relu2(self.bn2(self.conv(x1)))

        
        return x2
    
    
    
class up_convTD(nn.Module):
    
    """
    Perform upsampling in decoder section of U-Net. Used with time-distributed layers for time processing
    
    Params:
    
        - init: ch_in: input number of channels // ch_out: output number of channels
    
        - forward: x: input tensor
        
    """
    
    
    def __init__(self,ch_in,ch_out):
        
        super(up_convTD,self).__init__()
        
        self.up = TimeDistributed(nn.ConvTranspose2d(ch_in, ch_in, params.kernel_size, stride = 2,padding=params.padding, output_padding = 1))
        
        self.bn1 = TimeDistributed(nn.InstanceNorm2d(ch_in))

        self.relu1 = TimeDistributed(nn.LeakyReLU(inplace=True))
        
        self.conv = TimeDistributed(nn.Conv2d(ch_in,ch_out,kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True))
        
        self.bn2 = TimeDistributed(nn.InstanceNorm2d(ch_in))

        self.relu2 = TimeDistributed(nn.LeakyReLU(inplace=True))
        

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
    
    

class Attention_blockTD(nn.Module):
    
    """
    Time-distributed attention-block
    
    """
    
    def __init__(self,F_g,F_l,F_int):
        
        super(Attention_blockTD,self).__init__()
        
        self.W_g = TimeDistributed(nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True))
        
        self.bn1 = TimeDistributed(nn.InstanceNorm2d(F_int))
            
        
        self.W_x = TimeDistributed(nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True))
            
        self.bn2 = TimeDistributed(nn.InstanceNorm2d(F_int))
        
        

        self.psi = TimeDistributed(nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True))
        
        self.bn3 = TimeDistributed(nn.InstanceNorm2d(F_int))
        
        self.sigma = TimeDistributed(nn.Sigmoid())
        
        self.relu = TimeDistributed(nn.ReLU(inplace=True))
        
        
        
    def forward(self,g,x):
        
        g1 = self.bn1(self.W_g(g))
        
        x1 = self.bn2(self.W_x(x))
        
        psi = self.relu(g1+x1)
        
        psi = self.sigma(self.bn3(self.psi(psi)))

        return x*psi     
    
    
    
    
class Attention_block3d(nn.Module):
    
    def __init__(self,F_g,F_l,F_int):
        
        super(Attention_block3d,self).__init__()
        
        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        
        self.bn1 = nn.InstanceNorm3d(F_int)
            
        
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True)
            
        self.bn2 = nn.InstanceNorm3d(F_int)
        
        

        self.psi = nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True)
        
        self.bn3 = nn.InstanceNorm3d(1)
        
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
        
        self.pad = addRowCol2d()
        
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
        
        if params.supervision:
            
            self.Up_conv3_1 = nn.Conv2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), kernel_size=1,  stride=1,padding=0)
            
            self.Up_conv3_Transpose = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), len(params.class_weights), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
            
            self.Up_conv2_1 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), kernel_size=1,  stride=1,padding=0)
            
            self.Up_conv2_Transpose = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), params.kernel_size, stride = 1, padding = params.padding, output_padding = 0)
            
                           
                           
    def forward(self,x):
                           
        if params.three_D or (not(params.three_D) and params.add3d > 0):
                           
            # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)

            x = x.view(x.shape[0]*x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
                           
        # encoding path
                           
        x1 = self.Down1(x, 'encoder')

        x2 = self.Maxpool(x1)
        x2 = self.drop(self.Down2(x2, 'encoder'))
        
        x3 = self.Maxpool(x2)
        x3 = self.drop(self.Down3(x3, 'encoder'))
        
        
        d3 = self.Up3(x3)
        
        if (d3.shape[-1] != x2.shape[-1]) or (d3.shape[-2] != x2.shape[-2]):
            
            d3 = self.pad(d3, x2.shape)
            
        
        x2 = self.Att3(g=d3,x=x2)
        d3 = self.Up_conv3(self.cat(x2,d3), 'decoder') # Output to supervision

        d2 = self.Up2(d3)
        
        if d2.shape[-1] != x1.shape[-1] or (d3.shape[-2] != x2.shape[-2]):
            
            d2 = self.pad(d2, x1.shape)
        
        x1 = self.Att2(g=d2,x=x1)
        d2 = self.Up_conv2(self.cat(x1,d2), 'decoder')  # Output to supervision

        d1 = self.Conv_1x1(d2)
        
        if params.three_D or (not(params.three_D) and params.add3d > 0):
            
            d1 = d1.view(params.batch_size, d1.shape[1], d1.shape[2], d1.shape[3], d1.shape[0]//params.batch_size)
            
        if params.supervision:
            
            d3 = self.Up_conv3_1(d3)
            
            d3 = self.Up_conv3_Transpose(d3)
            
            d2 = self.Up_conv2_1(d2)
            
            d2 = self.Up_conv2_Transpose(d2)
            
        if params.test or not(params.supervision):

            return d1 
        
        else:
            
            return [d1,d2,d3]
                           
                           
class NewAttentionUNet(nn.Module):
    
    """
    U-Net with residuals architecture, extracted from Bratt et al., 2019 paper, including Attention Gates, from Oktay et al., 2018 paper. Does not include subsamplings nor upsamplings
    
    Processes 2D data
    
    """

    def __init__(self):
        
        super(NewAttentionUNet, self).__init__()

        self.cat = Concat()
        
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
                           
        x1 = self.Down1(x, 'encoder')

        x2 = self.drop(self.Down2(x1, 'encoder'))
        
        x3 = self.drop(self.Down3(x2, 'encoder'))
        
        d3 = self.Up_conv3(x3, 'decoder')
        
        x2 = self.Att3(g=d3,x=x2)
        
        d2 = self.Up_conv2(self.cat(x2[:,(params.base*(2**(params.num_layers - 3))):],d3[:,(params.base*(2**(params.num_layers - 3))):]), 'decoder')
        
        x1 = self.Att2(g=d2,x=x1)

        d1 = self.Conv_1x1(self.cat(x1[:,(int(params.base*(2**(params.num_layers - 4)))):],d2[:,(int(params.base*(2**(params.num_layers - 4)))):]))
        
        if params.three_D or (not(params.three_D) and params.add3d > 0):
            
            d1 = d1.view(params.batch_size, d1.shape[1], d1.shape[2], d1.shape[3], d1.shape[0]//params.batch_size)
            

        return d1                            

    
class TimeDistributedAttentionUNet(nn.Module):
    
    
    """
    Time-distributed version of U-Net with residuals architecture, extracted from Bratt et al., 2019 paper, including Attention Gates, from Oktay et al., 2018 paper. Allow for time information processing
    
    Processes 2D+time data
    
    """

    def __init__(self):
        
        super(TimeDistributedAttentionUNet, self).__init__()

        self.cat = ConcatTime()
        
        self.pad = addRowCol()
        
        self.Maxpool = TimeDistributed(nn.MaxPool2d(kernel_size=2,stride=2))
        
        # Decide on number of input channels
        
        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1

            
        # Encoder

        self.Down1 = conv_res_blockTD(ch_in=in_chan, ch_out=params.base*(2**(params.num_layers - 3)), key = 'encoder')
        
        self.Down2 = conv_res_blockTD(ch_in=params.base*(2**(params.num_layers - 3)),ch_out=params.base*(2**(params.num_layers - 2)), key = 'encoder')
    
        self.Down3 = conv_res_blockTD(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 1)), key = 'encoder')
        
        self.drop = TimeDistributed(nn.Dropout2d(params.dropout))
   
        
        # Decoder + Attention Gates                                           
                                                   
        self.Up3 = up_convTD(ch_in=params.base*(2**(params.num_layers - 1)),ch_out=params.base*(2**(params.num_layers - 2)))
                                                   
        self.Att3 = Attention_blockTD(F_g=params.base*(2**(params.num_layers - 2)),F_l=params.base*(2**(params.num_layers - 2)),F_int=params.base*(2**(params.num_layers - 3)))
                                                   
        self.Up_conv3 = conv_res_blockTD(ch_in=params.base*(2**(params.num_layers - 1)), ch_out=params.base*(2**(params.num_layers - 2)), key = 'decoder')
        
        self.Up2 = up_convTD(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 3)))
            
        self.Att2 = Attention_blockTD(F_g=params.base*(2**(params.num_layers - 3)),F_l=params.base*(2**(params.num_layers - 3)),F_int=int(params.base*(2**(params.num_layers - 4))))
                                                   
        self.Up_conv2 = conv_res_blockTD(ch_in=params.base*(2**(params.num_layers - 2)), ch_out=params.base*(2**(params.num_layers - 3)), key = 'decoder')

        self.Conv_1x1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
        
        if params.supervision:
            
            self.Up_conv3_1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), kernel_size=1,  stride=1,padding=0))
            
            self.Up_conv3_Transpose = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), len(params.class_weights), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
            
            self.Up_conv2_1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), kernel_size=1,  stride=1,padding=0))
            
            self.Up_conv2_Transpose = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), params.kernel_size, stride = 1, padding = params.padding, output_padding = 0))
                           
                           
    def forward(self,x):
                           
        # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)

        x = x.view(x.shape[0], x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
                           
        # encoding path
                           
        x1 = self.Down1(x, 'encoder')

        x2 = self.Maxpool(x1)
        x2 = self.drop(self.Down2(x2, 'encoder'))
        
        x3 = self.Maxpool(x2)
        x3 = self.drop(self.Down3(x3, 'encoder'))
        
        
        d3 = self.Up3(x3)
        
        if d3.shape[2] != x2.shape[2]:
            
            d3 = self.pad(d3)

        
        x2 = self.Att3(g=d3,x=x2)
        
        d3 = self.Up_conv3(self.cat(x2,d3), 'decoder')

        d2 = self.Up2(d3)
        
        if d2.shape[2] != x1.shape[2]:
            
            d2 = self.pad(d2)
        
        x1 = self.Att2(g=d2,x=x1)
        d2 = self.Up_conv2(self.cat(x1,d2), 'decoder')

        d1 = self.Conv_1x1(d2)
        
        # Reshape output: (BTCHW) --> (BCHWT)
            
        d1 = d1.view(params.batch_size, d1.shape[2], d1.shape[3], d1.shape[4], d1.shape[1])
        
        if params.supervision:
            
            d3 = self.Up_conv3_1(d3)
            
            d3 = self.Up_conv3_Transpose(d3)
            
            d2 = self.Up_conv2_1(d2)
            
            d2 = self.Up_conv2_Transpose(d2)
            
            d3 = d3.view(params.batch_size, d3.shape[2], d3.shape[3], d3.shape[4], d3.shape[1])
            
            d2 = d2.view(params.batch_size, d2.shape[2], d2.shape[3], d2.shape[4], d2.shape[1])

            
            
        if params.test or not(params.supervision):

            return d1 
        
        else:
            
            return [d1,d2,d3]

                       

        
class TimeDistributedUNet(nn.Module):
    
    
    """
    Time-distributed version of U-Net with residuals architecture, extracted from Bratt et al., 2019 paper, including recurrent units in the encoder-to-decoder skip connections, as in hourglass paper. Allow for time information processing
    
    Processes 2D+time data
    
    """

    def __init__(self):
        
        super(TimeDistributedUNet, self).__init__()

        self.cat = ConcatTime()
        
        self.pad = addRowCol()
        
        self.Maxpool = TimeDistributed(nn.MaxPool2d(kernel_size=2,stride=2))
        
        # Decide on number of input channels
        
        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1

            
        # Encoder

        self.Down1 = conv_res_blockTD(ch_in=in_chan, ch_out=params.base*(2**(params.num_layers - 3)), key = 'encoder')
        
        self.Down2 = conv_res_blockTD(ch_in=params.base*(2**(params.num_layers - 3)),ch_out=params.base*(2**(params.num_layers - 2)), key = 'encoder')
    
        self.Down3 = conv_res_blockTD(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 1)), key = 'encoder')
        
        self.drop = TimeDistributed(nn.Dropout2d(params.dropout))
        
        # Encoder-to-decoder recurrent connections
        
        if params.rnn == 'LSTM':

            self.rnn3 = ConvLSTM(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), (params.kernel_size, params.kernel_size), 1, True, True, False)
            
            self.rnn2 = ConvLSTM(params.base*(2**(params.num_layers - 2)), int(params.base*(2**(params.num_layers - 2))), (params.kernel_size, params.kernel_size), 1, True, True, False)

        elif params.rnn == 'GRU':

            if torch.cuda.is_available():

                dtype = torch.cuda.FloatTensor # computation in GPU

            else:

                dtype = torch.FloatTensor

            self.rnn3 = ConvGRU(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
            self.rnn2 = ConvGRU(params.base*(2**(params.num_layers - 2)), int(params.base*(2**(params.num_layers - 2))), (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
        else:
            
            print('Wrong recurrent cell. Please type "LSTM" or "GRU" as possible names for a recurrent cell')
   

        # Decoder                                            
                                                   
        self.Up3 = up_convTD(ch_in=params.base*(2**(params.num_layers - 1)),ch_out=params.base*(2**(params.num_layers - 2)))
                                           
        self.Up_conv3 = conv_res_blockTD(ch_in=params.base*(2**(params.num_layers - 1)), ch_out=params.base*(2**(params.num_layers - 2)), key = 'decoder')
        
        self.Up2 = up_convTD(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 3)))
                                                   
        self.Up_conv2 = conv_res_blockTD(ch_in=params.base*(2**(params.num_layers - 2)), ch_out=params.base*(2**(params.num_layers - 3)), key = 'decoder')

        self.Conv_1x1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
        
        if params.supervision:
            
            self.Up_conv3_1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), kernel_size=1,  stride=1,padding=0))
            
            self.Up_conv3_Transpose = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), len(params.class_weights), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
            
            self.Up_conv2_1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), kernel_size=1,  stride=1,padding=0))
            
            self.Up_conv2_Transpose = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), params.kernel_size, stride = 1, padding = params.padding, output_padding = 0))
                           
                           
    def forward(self,x):
                           
        # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)

        x = x.view(x.shape[0], x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
                           
        # encoding path
                           
        x1 = self.Down1(x, 'encoder')

        x2 = self.Maxpool(x1)
        x2 = self.drop(self.Down2(x2, 'encoder'))

        
        x3 = self.Maxpool(x2)
        x3 = self.drop(self.Down3(x3, 'encoder'))
        
        
        d3 = self.Up3(x3)
        
        if d3.shape[2] != x2.shape[2]:
            
            d3 = self.pad(d3)

        
        cat3 = self.cat(x2,d3)
        
        d3,_ = self.rnn3(cat3)
        
        d3 = self.Up_conv3(d3[0], 'decoder')

        d2 = self.Up2(d3)
        
        if d2.shape[2] != x1.shape[2]:
            
            d2 = self.pad(d2)
            
        cat2 = self.cat(x1,d2)
        
        d2,_ = self.rnn2(cat2)
        
        d2 = self.Up_conv2(d2[0], 'decoder')

        d1 = self.Conv_1x1(d2)
        
        # Reshape output: (BTCHW) --> (BCHWT)
            
        d1 = d1.view(params.batch_size, d1.shape[2], d1.shape[3], d1.shape[4], d1.shape[1])
        
        if params.supervision:
            
            d3 = self.Up_conv3_1(d3)
            
            d3 = self.Up_conv3_Transpose(d3)
            
            d2 = self.Up_conv2_1(d2)
            
            d2 = self.Up_conv2_Transpose(d2)
            
            d3 = d3.view(params.batch_size, d3.shape[2], d3.shape[3], d3.shape[4], d3.shape[1])
            
            d2 = d2.view(params.batch_size, d2.shape[2], d2.shape[3], d2.shape[4], d2.shape[1])
            
            
        if params.test or not(params.supervision):

            return d1 
        
        else:
            
            return [d1,d2,d3]

    

    
class conv_res_blockTD_nonrec(nn.Module):
    
    """
    Convolutional block with residuals for Attention U-Net.
    
    Can optionally be used with recurrent units, too
    
    Params:
    
    - Init: ch_in: number of input channels // ch_out: number of output channels // key: flag stating if the block is in the encoder or in the decoder
    
    - Forward: x: input tensor
    
    Outputs: x: output tensor
    
    
    """
    
    def __init__(self,ch_in,ch_out):
        
        super(conv_res_blockTD_nonrec,self).__init__()

        self.conv1 = TimeDistributed(nn.Conv2d(ch_in, ch_out, kernel_size=params.kernel_size, stride=1, padding=params.padding, bias=True))
        
        self.bn1 = TimeDistributed(nn.InstanceNorm2d(ch_out))
        
        self.relu1 = TimeDistributed(nn.ReLU(inplace=True))
        
        self.conv2 = TimeDistributed(nn.Conv2d(ch_out, ch_out, kernel_size=params.kernel_size,stride=1,padding=params.padding,bias=True))
        
        self.bn2 = TimeDistributed(nn.InstanceNorm2d(ch_out))

        self.relu2 = TimeDistributed(nn.ReLU(inplace=True))
        
        self.Conv_1x1 = TimeDistributed(nn.Conv2d(ch_in,ch_out,kernel_size=params.kernel_size,stride=1,padding=params.padding))


    def forward(self,x):
        
        x0 = self.Conv_1x1(x)

        x1 = self.relu1(self.bn1(self.conv1(x)))
        
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        
        return x0 + x2
    
    
                       

class Hourglass(nn.Module):
    
    
    """
    Time-distributed version of U-Net with residuals architecture, extracted from Bratt et al., 2019 paper, including recurrent units in the encoder-to-decoder skip connections, as in hourglass paper. Allow for time information processing
    
    Processes 2D+time data
    
    """

    def __init__(self):
        
        super(Hourglass, self).__init__()

        self.cat = ConcatTime()
        
        self.pad = addRowCol()
        
        self.Maxpool = TimeDistributed(nn.MaxPool2d(kernel_size=2,stride=2))
        
        # Decide on number of input channels
        
        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1

            
        # Encoder

        self.Down1 = conv_res_blockTD_nonrec(ch_in=in_chan, ch_out=params.base*(2**(params.num_layers - 3)))
        
        self.Down2 = conv_res_blockTD_nonrec(ch_in=params.base*(2**(params.num_layers - 3)),ch_out=params.base*(2**(params.num_layers - 2)))
    
        self.Down3 = conv_res_blockTD_nonrec(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 1)))
        
        self.drop = TimeDistributed(nn.Dropout2d(params.dropout))
        
        # Encoder-to-decoder recurrent connections
        
        if params.rnn == 'LSTM':

            self.rnn3 = ConvLSTM(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), (params.kernel_size, params.kernel_size), 1, True, True, False)
            
            self.rnn2 = ConvLSTM(params.base*(2**(params.num_layers - 2)), int(params.base*(2**(params.num_layers - 2))), (params.kernel_size, params.kernel_size), 1, True, True, False)

        elif params.rnn == 'GRU':

            if torch.cuda.is_available():

                dtype = torch.cuda.FloatTensor # computation in GPU

            else:

                dtype = torch.FloatTensor

            self.rnn3 = ConvGRU(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
            self.rnn2 = ConvGRU(params.base*(2**(params.num_layers - 2)), int(params.base*(2**(params.num_layers - 2))), (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)
            
        else:
            
            print('Wrong recurrent cell. Please type "LSTM" or "GRU" as possible names for a recurrent cell')
   

        # Decoder
                                                   
        self.Up3 = up_convTD(ch_in=params.base*(2**(params.num_layers - 1)),ch_out=params.base*(2**(params.num_layers - 2)))
        
        self.Up_conv3 = conv_res_blockTD_nonrec(ch_in=params.base*(2**(params.num_layers - 1)), ch_out=params.base*(2**(params.num_layers - 2)))
        
        self.Up2 = up_convTD(ch_in=params.base*(2**(params.num_layers - 2)),ch_out=params.base*(2**(params.num_layers - 3)))
                                                   
        self.Up_conv2 = conv_res_blockTD_nonrec(ch_in=params.base*(2**(params.num_layers - 2)), ch_out=params.base*(2**(params.num_layers - 3)))

        self.Conv_1x1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
        
        if params.supervision:
            
            self.Up_conv3_1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), kernel_size=1,  stride=1,padding=0))
            
            self.Up_conv3_Transpose = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), len(params.class_weights), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
            
            self.Up_conv2_1 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), kernel_size=1,  stride=1,padding=0))
            
            self.Up_conv2_Transpose = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), params.kernel_size, stride = 1, padding = params.padding, output_padding = 0))
                           
                           
    def forward(self,x):
                           
        # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)

        x = x.view(x.shape[0], x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
                           
        # Encoding path
                           
        x1 = self.Down1(x)

        x2 = self.Maxpool(x1)
        x2 = self.drop(self.Down2(x2))

        x3 = self.Maxpool(x2)
        x3 = self.drop(self.Down3(x3))

        
        # Decoding path
        
        d3 = self.Up3(x3)
        
        cat3 = self.cat(x2, d3)
        
        rnn3,_ = self.rnn3(cat3)
        
        d3 = self.Up_conv3(rnn3[0])
       
        d2 = self.Up2(d3)

        cat2 = self.cat(x1,d2)
        
        rnn2,_ = self.rnn2(cat2)
        
        d2 = self.Up_conv2(rnn2[0])

        d1 = self.Conv_1x1(d2)
        
        # Reshape output: (BTCHW) --> (BCHWT)
            
        d1 = d1.view(params.batch_size, d1.shape[2], d1.shape[3], d1.shape[4], d1.shape[1])

        if params.supervision:
            
            d3 = self.Up_conv3_1(d3)
            
            d3 = self.Up_conv3_Transpose(d3)
            
            d2 = self.Up_conv2_1(d2)
            
            d2 = self.Up_conv2_Transpose(d2)
            
            d3 = d3.view(params.batch_size, d3.shape[2], d3.shape[3], d3.shape[4], d3.shape[1])
            
            d2 = d2.view(params.batch_size, d2.shape[2], d2.shape[3], d2.shape[4], d2.shape[1])
            
            
        if params.test or not(params.supervision):

            return d1 
        
        else:
            
            return [d1,d2,d3]
                           


class convBlock2d(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels):
        
        super(convBlock2d, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        
        self.bn1 = nn.InstanceNorm2d(middle_channels)
        
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        
        self.bn2 = nn.InstanceNorm2d(out_channels)
        

    def forward(self, x):
        
        out = self.conv1(x)
        
        out = self.bn1(out)
        
        out = self.relu(out)

        out = self.conv2(out)
        
        out = self.bn2(out)
        
        out = self.relu(out)

        return out
    
    
class convBlock2dtime(nn.Module):
    
    """
    Convolutional block with time-distributed layers for time data processing
    
    Can optionally be used with recurrent units, too
    
    Params:
    
    - Init: ch_in: number of input channels // ch_out: number of output channels // key: flag stating if the block is in the encoder or in the decoder
    
    - Forward: x: input tensor
    
    Outputs: x: output tensor
    
    
    """
    
    def __init__(self,ch_in,ch_mid,ch_out, key):
        
        super(convBlock2dtime,self).__init__()
            
        if params.rnn_position == 'full' or params.rnn_position == key:

            if params.rnn == 'LSTM':

                self.conv1 = ConvLSTM(ch_in, ch_mid, (params.kernel_size , params.kernel_size), 1, True, True, False)

            elif params.rnn == 'GRU':

                if torch.cuda.is_available():

                    dtype = torch.cuda.FloatTensor # computation in GPU

                else:

                    dtype = torch.FloatTensor

                self.conv1 = ConvGRU(ch_in, ch_mid, (params.kernel_size, params.kernel_size), 1, dtype, True, True, False)

        else:

            self.conv1 = TimeDistributed(nn.Conv2d(ch_in, ch_mid, kernel_size=params.kernel_size, stride=1, padding=params.padding, bias=True))
        
        self.bn1 = TimeDistributed(nn.InstanceNorm2d(ch_mid))
        
        self.relu1 = TimeDistributed(nn.ReLU(inplace=True))
        
        self.conv2 = TimeDistributed(nn.Conv2d(ch_mid, ch_out, kernel_size=params.kernel_size, stride=1, padding=params.padding, bias=True))
        
        self.bn2 = TimeDistributed(nn.InstanceNorm2d(ch_out))

        self.relu2 = TimeDistributed(nn.ReLU(inplace=True))


    def forward(self,x, key):
        
        if params.rnn_position == 'full' or params.rnn_position == key:
        
            x,_ = list(self.conv1(x))
            
            x1 = self.relu1(self.bn1(x[0]))
            
        else:
        
            x1 = self.relu1(self.bn1(self.conv1(x)))
        
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        
        return x2
    
    
    
    
class NestedUNet2d(nn.Module):
    
    """
    Nested UNet from Zhou 2018
    
    """
    
    def __init__(self):
    
        super(NestedUNet2d, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Provide number of input channels
        
        if params.sum_work and 'both' in params.train_with:

            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time

        elif (params.sum_work and not('both' in params.train_with)):

            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time

        elif (not(params.sum_work) and 'both' in params.train_with):

            in_chan = 2 # Train with magnitude + phase (no sum)

        elif not(params.sum_work) and not('both' in params.train_with):

            in_chan = 1 # Train magnitude or phase (no sum)
            
            
        
        self.conv0_0 = convBlock2d(in_chan, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_0 = convBlock2d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)))
        
        self.conv2_0 = convBlock2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)))
        
        
        
        self.conv0_1 = convBlock2d(params.base*(2**(params.num_layers - 3)) + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_1 = convBlock2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)))
        
        
        self.conv0_2 = convBlock2d(params.base*(2**(params.num_layers - 2)) + params.base*(2**(params.num_layers - 3))*2, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_2 = convBlock2d(params.base*(2**(params.num_layers - 2))*2 + params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2))) 
        
        self.conv0_3 = convBlock2d(params.base*(2**(params.num_layers - 3))*3 + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        
        self.up10_01 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up11_02 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up12_03 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up20_12 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.Conv_1x1_03 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
        
        if params.supervision:
            
            self.Conv_1x1_02 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
            
            self.Conv_1x1_01 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)

       
    def forward(self, x):
        
        x_00 = self.conv0_0(x)
        
        x_10 = self.conv1_0(self.pool(x_00))
        
        x_20 = self.conv2_0(self.pool(x_10))
        
        x_01 = self.conv0_1(torch.cat((self.up10_01(x_10),x_00), dim = 1))
        
        x_11 = self.conv1_1(x_10)
        
        x_02 = self.conv0_2(torch.cat((self.up11_02(x_11), x_00, x_01),dim = 1))
        
        x_12 = self.conv1_2(torch.cat((self.up20_12(x_20), x_10, x_11),dim = 1))
        
        x_03 = self.conv0_3(torch.cat((self.up12_03(x_12), x_00, x_01, x_02), dim = 1))
        
        x_03 = self.Conv_1x1_03(x_03)
        
        if params.supervision and not(params.test):
            
            x_01 = self.Conv_1x1_01(x_01) 
            
            x_02 = self.Conv_1x1_02(x_02)
            
            return [x_03, x_02, x_01]
        
        else:
            
            return x_03
        
        
class NestedUNet2dTime(nn.Module):
    
    """
    Nested UNet from Zhou 2018, adapted to work in 2D+time
    
    """
    
    
    def __init__(self):
        
        super(NestedUNet2dTime, self).__init__()
        
        self.pool = TimeDistributed(nn.MaxPool2d(2, 2))
        
        # Decide on number of input channels
        
        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1

            
        self.conv0_0 = convBlock2dtime(in_chan, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'encoder')

        self.conv1_0 = convBlock2dtime(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'encoder')

        self.conv2_0 = convBlock2dtime(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), 'encoder')
            
        
        self.conv1_2 = convBlock2dtime(params.base*(2**(params.num_layers - 2))*2 + params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'decoder') 
        
        self.conv0_3 = convBlock2dtime(params.base*(2**(params.num_layers - 3))*3 + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'decoder')
        
            
        self.conv0_1 = convBlock2dtime(params.base*(2**(params.num_layers - 3)) + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'full')
        
        self.conv1_1 = convBlock2dtime(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'full')
        
        self.conv0_2 = convBlock2dtime(params.base*(2**(params.num_layers - 2)) + params.base*(2**(params.num_layers - 3))*2, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'full')    
            
        self.up10_01 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up11_02 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up12_03 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up20_12 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.Conv_1x1_03 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
        
        
        if params.supervision:
            
            self.Conv_1x1_02 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
            
            self.Conv_1x1_01 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
            
        
    def forward(self, x):
        
        # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)

        x = x.view(x.shape[0], x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
        
        x_00 = self.conv0_0(x, 'encoder')
        
        x_10 = self.conv1_0(self.pool(x_00), 'encoder')
        
        x_20 = self.conv2_0(self.pool(x_10), 'encoder')
        
        x_01_input = torch.cat((self.up10_01(x_10),x_00), dim = 2)
        
        x_01 = self.conv0_1(x_01_input, 'full')
        
        x_11 = self.conv1_1(x_10, 'full')
        
        x_02_input = torch.cat((self.up11_02(x_11), x_00, x_01),dim = 2)
        
        x_02 = self.conv0_2(x_02_input, 'full')
        
        x_12_input = torch.cat((self.up20_12(x_20), x_10, x_11),dim = 2)
        
        x_12 = self.conv1_2(x_12_input, 'decoder')
        
        x_03_input = torch.cat((self.up12_03(x_12), x_00, x_01, x_02), dim = 2)
        
        x_03 = self.conv0_3(x_03_input, 'decoder')
        
        x_03 = self.Conv_1x1_03(x_03)
        
        x_03 = x_03.view(params.batch_size, x_03.shape[2], x_03.shape[3], x_03.shape[4], x_03.shape[1])
        
        if params.supervision and not(params.test):
            
            x_01 = self.Conv_1x1_01(x_01) 
            
            x_02 = self.Conv_1x1_01(x_02)
            
            x_01 = x_01.view(params.batch_size, x_01.shape[2], x_01.shape[3], x_01.shape[4], x_01.shape[1])
            
            x_02 = x_02.view(params.batch_size, x_02.shape[2], x_02.shape[3], x_02.shape[4], x_02.shape[1])
            
            return [x_03, x_02, x_01]
        
        else:
            
            return x_03  
        
        
        
        
        
class NestedUNet2dAutocontext(nn.Module):
    
    """
    Nested UNet from Zhou 2018, with autocontext iterations (cascaded U-Net)
    
    """
    
    def __init__(self):
    
        super(NestedUNet2dAutocontext, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Provide number of input channels
        
        if params.sum_work and 'both' in params.train_with:

            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time

        elif (params.sum_work and not('both' in params.train_with)):

            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time

        elif (not(params.sum_work) and 'both' in params.train_with):

            in_chan = 2 # Train with magnitude + phase (no sum)

        elif not(params.sum_work) and not('both' in params.train_with):

            in_chan = 1 # Train magnitude or phase (no sum)
            
            

        
        self.conv0_0 = convBlock2d(in_chan + 1, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_0 = convBlock2d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)))
        
        self.conv2_0 = convBlock2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)))
        
        
        
        self.conv0_1 = convBlock2d(params.base*(2**(params.num_layers - 3)) + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_1 = convBlock2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)))
        
        
        self.conv0_2 = convBlock2d(params.base*(2**(params.num_layers - 2)) + params.base*(2**(params.num_layers - 3))*2, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_2 = convBlock2d(params.base*(2**(params.num_layers - 2))*2 + params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2))) 
        
        self.conv0_3 = convBlock2d(params.base*(2**(params.num_layers - 3))*3 + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        
        self.up10_01 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up11_02 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up12_03 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up20_12 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.Conv_1x1_03 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
        
        if params.supervision:
            
            self.Conv_1x1_02 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
            
            self.Conv_1x1_01 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)

       
    def forward(self, x):
        
        for i in range(params.autocontext_iter):
            
            if i == 0:
        
                x_out = torch.zeros(x.shape[0],1,x.shape[-2],x.shape[-1]) + 0.5
            
                x_out = x_out.cuda()
            
            else:
                
                x_out = torch.argmax(x_03, 1).cuda().float()
                
                x_out += 0.5
                
                x_out = x_out.view(x.shape[0],1,x.shape[-2],x.shape[-1])
                
            x_in = torch.cat([x,x_out], dim=1)
                
            x_00 = self.conv0_0(x_in)

            x_10 = self.conv1_0(self.pool(x_00))

            x_20 = self.conv2_0(self.pool(x_10))

            x_01 = self.conv0_1(torch.cat((self.up10_01(x_10),x_00), dim = 1))

            x_11 = self.conv1_1(x_10)

            x_02 = self.conv0_2(torch.cat((self.up11_02(x_11), x_00, x_01),dim = 1))

            x_12 = self.conv1_2(torch.cat((self.up20_12(x_20), x_10, x_11),dim = 1))

            x_03 = self.conv0_3(torch.cat((self.up12_03(x_12), x_00, x_01, x_02), dim = 1))

            x_03 = self.Conv_1x1_03(x_03)
                
            if params.supervision and not(params.test) and i == params.autocontext_iter - 1:

                x_01 = self.Conv_1x1_01(x_01) 

                x_02 = self.Conv_1x1_01(x_02)

                return [x_03, x_02, x_01]

            elif (not(params.supervision) and i == params.autocontext_iter - 1) or (params.test and i == params.autocontext_iter - 1):

                return x_03

            
            
class NestedUNet2dTimeAutocontext(nn.Module):
    
    """
    Nested UNet from Zhou 2018, adapted to work in 2D+time, including autocontext
    
    """
    
    
    def __init__(self):
        
        super(NestedUNet2dTimeAutocontext, self).__init__()
        
        self.pool = TimeDistributed(nn.MaxPool2d(2, 2))
        
        # Decide on number of input channels
        
        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1

            
        self.conv0_0 = convBlock2dtime(in_chan + 1, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'encoder')

        self.conv1_0 = convBlock2dtime(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'encoder')

        self.conv2_0 = convBlock2dtime(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), 'encoder')
            
        
        self.conv1_2 = convBlock2dtime(params.base*(2**(params.num_layers - 2))*2 + params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'decoder') 
        
        self.conv0_3 = convBlock2dtime(params.base*(2**(params.num_layers - 3))*3 + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'decoder')
        
            
        self.conv0_1 = convBlock2dtime(params.base*(2**(params.num_layers - 3)) + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'full')
        
        self.conv1_1 = convBlock2dtime(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'full')
        
        self.conv0_2 = convBlock2dtime(params.base*(2**(params.num_layers - 2)) + params.base*(2**(params.num_layers - 3))*2, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'full')    
            
        self.up10_01 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up11_02 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up12_03 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up20_12 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.Conv_1x1_03 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
        
        
        if params.supervision:
            
            self.Conv_1x1_02 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
            
            self.Conv_1x1_01 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
            
        
    def forward(self, x):
        
        for i in range(params.autocontext_iter):
            
            if i == 0:
        
                x_out = torch.zeros(x.shape[0],1,x.shape[-3],x.shape[-2],x.shape[-1]) + 0.5
            
                x_out = x_out.cuda()
            
            else:
                
                x_out = torch.argmax(x_03, 1).cuda().float()
                
                x_out += 0.5
                
                x_out = x_out.view(x.shape[0],1,x.shape[-3], x.shape[-2],x.shape[-1])
                
            x_in = torch.cat([x,x_out], dim=1)
        
            # Reshape input: (B, C, H, W, T) --> (B, T, C, H, W)

            x_in = x_in.view(x_in.shape[0], x_in.shape[-1], x_in.shape[1], x_in.shape[-3], x_in.shape[-2])

            x_00 = self.conv0_0(x_in, 'encoder')

            x_10 = self.conv1_0(self.pool(x_00), 'encoder')

            x_20 = self.conv2_0(self.pool(x_10), 'encoder')

            x_01_input = torch.cat((self.up10_01(x_10),x_00), dim = 2)

            x_01 = self.conv0_1(x_01_input, 'full')

            x_11 = self.conv1_1(x_10, 'full')

            x_02_input = torch.cat((self.up11_02(x_11), x_00, x_01),dim = 2)

            x_02 = self.conv0_2(x_02_input, 'full')

            x_12_input = torch.cat((self.up20_12(x_20), x_10, x_11),dim = 2)

            x_12 = self.conv1_2(x_12_input, 'decoder')

            x_03_input = torch.cat((self.up12_03(x_12), x_00, x_01, x_02), dim = 2)

            x_03 = self.conv0_3(x_03_input, 'decoder')

            x_03 = self.Conv_1x1_03(x_03)

            x_03 = x_03.view(params.batch_size, x_03.shape[2], x_03.shape[3], x_03.shape[4], x_03.shape[1])

            if params.supervision and not(params.test) and i == params.autocontext_iter - 1:

                x_01 = self.Conv_1x1_01(x_01) 

                x_02 = self.Conv_1x1_01(x_02)

                x_01 = x_01.view(params.batch_size, x_01.shape[2], x_01.shape[3], x_01.shape[4], x_01.shape[1])

                x_02 = x_02.view(params.batch_size, x_02.shape[2], x_02.shape[3], x_02.shape[4], x_02.shape[1])

                return [x_03, x_02, x_01]

            elif (not(params.supervision) and i == params.autocontext_iter - 1) or (params.test and i == params.autocontext_iter - 1):

                return x_03  
            
            
            
            
            
class NestedUNet2dScale(nn.Module):
    
    
    """
    Nested U-Net from Zhou, 2018 with two stacked units. One for coarse scale analysis and vessel location and another one for fine scale analysis and definitive segmentation
    
    """
    
    
    def __init__(self):
        
        super(NestedUNet2dScale, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Provide number of input channels
        
        if params.sum_work and 'both' in params.train_with:

            in_chan = 7 # Train with magnitude + phase + sum of both along time + MIP of both along time

        elif (params.sum_work and not('both' in params.train_with)):

            in_chan = 3 # Train with magnitude or phase, sum in time and MIP in time

        elif (not(params.sum_work) and 'both' in params.train_with):

            in_chan = 2 # Train with magnitude + phase (no sum)

        elif not(params.sum_work) and not('both' in params.train_with):

            in_chan = 1 # Train magnitude or phase (no sum)
            
            
        self.pad = addRowCol2d()
        
        
        self.conv0_0 = convBlock2d(in_chan, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_0 = convBlock2d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)))
        
        self.conv2_0 = convBlock2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)))
        
        
        
        self.conv0_1 = convBlock2d(params.base*(2**(params.num_layers - 3)) + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_1 = convBlock2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)))
        
        
        self.conv0_2 = convBlock2d(params.base*(2**(params.num_layers - 2)) + params.base*(2**(params.num_layers - 3))*2, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_2 = convBlock2d(params.base*(2**(params.num_layers - 2))*2 + params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2))) 
        
        self.conv0_3 = convBlock2d(params.base*(2**(params.num_layers - 3))*3 + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        
        self.up10_01 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up11_02 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up12_03 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up20_12 = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.Conv_1x1_03 = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
        
        
        
        self.conv0_0_new = convBlock2d(in_chan, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_0_new = convBlock2d(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)))
        
        self.conv2_0_new = convBlock2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)))
        
        
        
        self.conv0_1_new = convBlock2d(params.base*(2**(params.num_layers - 3)) + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_1_new = convBlock2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)))
        
        
        self.conv0_2_new = convBlock2d(params.base*(2**(params.num_layers - 2)) + params.base*(2**(params.num_layers - 3))*2, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        self.conv1_2_new = convBlock2d(params.base*(2**(params.num_layers - 2))*2 + params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2))) 
        
        self.conv0_3_new = convBlock2d(params.base*(2**(params.num_layers - 3))*3 + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)))
        
        
        self.up10_01_new = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up11_02_new = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up12_03_new = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.up20_12_new = nn.ConvTranspose2d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1)
        
        self.Conv_1x1_03_new = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
        
        
        if params.supervision:
            
            self.Conv_1x1_02_new = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)
            
            self.Conv_1x1_01_new = nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0)

       
    def forward(self, x):
        
        x_00 = self.conv0_0(x)
        
        x_10 = self.conv1_0(self.pool(x_00))
        
        x_20 = self.conv2_0(self.pool(x_10))
        
        x_01 = self.conv0_1(torch.cat((self.up10_01(x_10),x_00), dim = 1))
        
        x_11 = self.conv1_1(x_10)
        
        x_02 = self.conv0_2(torch.cat((self.up11_02(x_11), x_00, x_01),dim = 1))
        
        x_12 = self.conv1_2(torch.cat((self.up20_12(x_20), x_10, x_11),dim = 1))
        
        x_03 = self.conv0_3(torch.cat((self.up12_03(x_12), x_00, x_01, x_02), dim = 1))
        
        x_03 = self.Conv_1x1_03(x_03)
        
        
        x_inter, pads, final_centers, diff_values = utilities.connectedComponentsModule(x_03, x)
        
        x_inter_array = x_inter.cpu().detach().numpy()
        
        x_00_new = self.conv0_0_new(x_inter)
        
        x_10_new = self.conv1_0_new(self.pool(x_00_new))
        
        x_20_new = self.conv2_0_new(self.pool(x_10_new))
        
        if (self.up10_01_new(x_10_new).shape[-2] != x_00_new.shape[-2]) or (self.up10_01_new(x_10_new).shape[-1] != x_00_new.shape[-1]):
            
            up10_01_new = self.pad(self.up10_01_new(x_10_new), x_00_new.shape)
            
        else:
            
            up10_01_new = self.up10_01_new(x_10_new).clone()
        
        x_01_new = self.conv0_1_new(torch.cat((up10_01_new,x_00_new), dim = 1))
        
        x_11_new = self.conv1_1_new(x_10_new)
        
        if (self.up11_02_new(x_11_new).shape[-2] != x_00_new.shape[-2]) or (self.up11_02_new(x_11_new).shape[-1] != x_00_new.shape[-1]):
            
            up11_02_new = self.pad(self.up11_02_new(x_11_new), x_00_new.shape)
            
        else:
            
            up11_02_new = self.up11_02_new(x_11_new).clone()
        
        x_02_new = self.conv0_2_new(torch.cat((up11_02_new, x_00_new, x_01_new),dim = 1))
        
        if (self.up11_02_new(x_11_new).shape[-2] != x_10_new.shape[-2]) or (self.up11_02_new(x_11_new).shape[-1] != x_10_new.shape[-1]):
            
            up20_12_new = self.pad(self.up20_12_new(x_20_new), x_10_new.shape)
            
        else:
            
            up20_12_new = self.up20_12_new(x_20_new).clone()
        
        x_12_new = self.conv1_2_new(torch.cat((up20_12_new, x_10_new, x_11_new),dim = 1))
        
        if (self.up12_03_new(x_12_new).shape[-2] != x_00_new.shape[-2]) or (self.up12_03_new(x_12_new).shape[-1] != x_00_new.shape[-1]):
            
            up12_03_new = self.pad(self.up12_03_new(x_12_new), x_00_new.shape)
            
        else:
            
            up12_03_new = self.up12_03_new(x_12_new).clone()
        
        x_03_new = self.conv0_3_new(torch.cat((up12_03_new, x_00_new, x_01_new, x_02_new), dim = 1))
        
        x_03_new = self.Conv_1x1_03_new(x_03_new)
        
        x_03_out = utilities.padding(x_03_new, pads)
        
        if params.supervision and not(params.test):
            
            x_01_new = self.Conv_1x1_01_new(x_01_new) 
            
            x_02_new = self.Conv_1x1_02_new(x_02_new)
            
            x_01_out = utilities.padding(x_01_new, pads) 
            
            x_02_out = utilities.padding(x_02_new, pads)
            
            return [x_03_new, x_02_new, x_01_new], final_centers, diff_values, pads
        
        elif params.test:
            
            return x_03_out
        
        else:
            
            return x_03_new, final_centers, diff_values, pads
        
        
        
        
class NestedUNet2dTimeScale(nn.Module):
    
    """
    Nested UNet from Zhou 2018, adapted to work in 2D+time. In a second iteration, works at a finer scale
    
    """
    
    
    def __init__(self):
        
        super(NestedUNet2dTimeScale, self).__init__()
        
        self.pool = TimeDistributed(nn.MaxPool2d(2, 2))
        
        # Decide on number of input channels
        
        if 'both' in params.train_with:

            in_chan = 2

        else:

            in_chan = 1

            
        self.conv0_0 = convBlock2dtime(in_chan, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'encoder')

        self.conv1_0 = convBlock2dtime(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'encoder')

        self.conv2_0 = convBlock2dtime(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), 'encoder')
            
        
        self.conv1_2 = convBlock2dtime(params.base*(2**(params.num_layers - 2))*2 + params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'decoder') 
        
        self.conv0_3 = convBlock2dtime(params.base*(2**(params.num_layers - 3))*3 + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'decoder')
        
            
        self.conv0_1 = convBlock2dtime(params.base*(2**(params.num_layers - 3)) + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'full')
        
        self.conv1_1 = convBlock2dtime(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'full')
        
        self.conv0_2 = convBlock2dtime(params.base*(2**(params.num_layers - 2)) + params.base*(2**(params.num_layers - 3))*2, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'full')    
            
        self.up10_01 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up11_02 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up12_03 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up20_12 = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.Conv_1x1_03 = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
        
        
        
        
        self.conv0_0_new = convBlock2dtime(in_chan, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'encoder')

        self.conv1_0_new = convBlock2dtime(params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'encoder')

        self.conv2_0_new = convBlock2dtime(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), 'encoder')
            
        
        self.conv1_2_new = convBlock2dtime(params.base*(2**(params.num_layers - 2))*2 + params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'decoder') 
        
        self.conv0_3_new = convBlock2dtime(params.base*(2**(params.num_layers - 3))*3 + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'decoder')
        
            
        self.conv0_1_new = convBlock2dtime(params.base*(2**(params.num_layers - 3)) + params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'full')
        
        self.conv1_1_new = convBlock2dtime(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), 'full')
        
        self.conv0_2_new = convBlock2dtime(params.base*(2**(params.num_layers - 2)) + params.base*(2**(params.num_layers - 3))*2, params.base*(2**(params.num_layers - 3)), params.base*(2**(params.num_layers - 3)), 'full')    
            
        self.up10_01_new = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up11_02_new = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up12_03_new = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 2)), params.base*(2**(params.num_layers - 2)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.up20_12_new = TimeDistributed(nn.ConvTranspose2d(params.base*(2**(params.num_layers - 1)), params.base*(2**(params.num_layers - 1)), params.kernel_size, stride = 2, padding = params.padding, output_padding = 1))
        
        self.Conv_1x1_03_new = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
        
        
        if params.supervision:
            
            self.Conv_1x1_02_new = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
            
            self.Conv_1x1_01_new = TimeDistributed(nn.Conv2d(params.base*(2**(params.num_layers - 3)), len(params.class_weights), kernel_size=1, stride=1,padding=0))
            
        
    def forward(self, x):
        
        # Reshape input: (B, C, H, W, T) --> (B*T, C, H, W)

        x_v = x.view(x.shape[0], x.shape[-1], x.shape[1], x.shape[-3], x.shape[-2])
        
        x_00 = self.conv0_0(x_v, 'encoder')
        
        x_10 = self.conv1_0(self.pool(x_00), 'encoder')
        
        x_20 = self.conv2_0(self.pool(x_10), 'encoder')
        
        x_01_input = torch.cat((self.up10_01(x_10),x_00), dim = 2)
        
        x_01 = self.conv0_1(x_01_input, 'full')
        
        x_11 = self.conv1_1(x_10, 'full')
        
        x_02_input = torch.cat((self.up11_02(x_11), x_00, x_01),dim = 2)
        
        x_02 = self.conv0_2(x_02_input, 'full')
        
        x_12_input = torch.cat((self.up20_12(x_20), x_10, x_11),dim = 2)
        
        x_12 = self.conv1_2(x_12_input, 'decoder')
        
        x_03_input = torch.cat((self.up12_03(x_12), x_00, x_01, x_02), dim = 2)
        
        x_03 = self.conv0_3(x_03_input, 'decoder')
        
        x_03 = self.Conv_1x1_03(x_03)
        
        x_03 = x_03.view(params.batch_size, x_03.shape[2], x_03.shape[3], x_03.shape[4], x_03.shape[1])
        
        x_inter, pads, final_centers, diff_values = utilities.connectedComponentsModule(x_03, x)
        
        
        x_00_new = self.conv0_0_new(x_v, 'encoder')
        
        x_10_new = self.conv1_0_new(self.pool(x_00_new), 'encoder')
        
        x_20_new = self.conv2_0_new(self.pool(x_10_new), 'encoder')
        
        x_01_input_new = torch.cat((self.up10_01_new(x_10_new),x_00_new), dim = 2)
        
        x_01_new = self.conv0_1_new(x_01_input_new, 'full')
        
        x_11_new = self.conv1_1_new(x_10_new, 'full')
        
        x_02_input_new = torch.cat((self.up11_02_new(x_11_new), x_00_new, x_01_new),dim = 2)
        
        x_02_new = self.conv0_2_new(x_02_input_new, 'full')
        
        x_12_input_new = torch.cat((self.up20_12_new(x_20_new), x_10_new, x_11_new),dim = 2)
        
        x_12_new = self.conv1_2_new(x_12_input_new, 'decoder')
        
        x_03_input_new = torch.cat((self.up12_03_new(x_12_new), x_00_new, x_01_new, x_02_new), dim = 2)
        
        x_03_new = self.conv0_3_new(x_03_input_new, 'decoder')
        
        x_03_new = self.Conv_1x1_03_new(x_03_new)
        
        x_03_new = x_03_new.view(params.batch_size, x_03_new.shape[2], x_03_new.shape[3], x_03_new.shape[4], x_03_new.shape[1])

        
        
        
        if params.supervision and not(params.test):
            
            x_01 = self.Conv_1x1_01(x_01) 
            
            x_02 = self.Conv_1x1_01(x_02)
            
            x_01 = x_01.view(params.batch_size, x_01.shape[2], x_01.shape[3], x_01.shape[4], x_01.shape[1])
            
            x_02 = x_02.view(params.batch_size, x_02.shape[2], x_02.shape[3], x_02.shape[4], x_02.shape[1])
            
            return [x_03, x_02, x_01]
        
        else:
            
            return x_03  