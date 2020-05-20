#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:11:02 2020

@author: andres
"""

import math

import numpy as np

# Data folders

excel_path = '/home/andres/Documents/_Data/CKD_Part1/'

excel_file = 'CKD_QFlow_results.xlsx' 

raw_path_ckd1 = '/home/andres/Documents/_Data/_Patients/_Raw/_ckd1/'

# CKD flow measurements in CKD Part 2

raw_path_ckd2 = '/home/andres/Documents/_Data/CKD_Part2/4_Flow/'

study2 = 'CKD2'

raw_path_hero = '/home/andres/Documents/_Data/Heroic/_Flow/'

studyh = 'Hero'

raw_path_extr = '/home/andres/Documents/_Data/Extra/_Flow/'

studye = 'Extr'



flow_folders = [raw_path_ckd2, raw_path_hero, raw_path_extr] # Folders with flow information for patient stratification

studies_flow = ['CKD2', 'Hero', 'Extr'] # Studies with flow information

#flow_folders = [raw_path_ckd2, raw_path_extr] # Folders with flow information for patient stratification

#studies_flow = ['CKD2', 'Extr'] # Studies with flow information



rep = True # Repetition of minority studies in data presented to the network 


# Data selection parameters

prep_step = 'crop' # Level of preprocessing to apply for images to the network

train_with = 'magBF' # Type of images to train with

three_D = False # Train with separate 2D slices or 2D + time volumes

add3d = 2 # Number of past and future neighboring slices to build a 2.5D dataset

sum_work = False # If True, include an extra channel with the sum of all frames along time, else dont (only in 2D)

channel_count = 1 # Channel counter

multi_view = False # If True, allow for a multi-view analysis along time

supervision = True # If True, perform deep supervision on given architecture

test = False # State if model is being trained (False) or tested/validated (True)

autocontext = False # Cascaded network

autocontext_iter = 2 # Autocontext iterations

# Augmentation parameters

augmentation = True

augm_params = [0.15, 10] # Augmentation limit parameters: maximum mean noise amplitude, maximum scale and maximum degree rotation

augm_probs = [0.5]*5 # Probabilities for each augmentation event

# Optional 2D augmentation with albumentations

augm2D_limits = [0.95, 10] # Augmentation limit parameters: Scale, Rotation, Flip

augm2D_probs = [0.5]*5

# Cross-validation parameters

k = 7 # Number of cross validation folds

# Architecture parameters

rnn = 'GRU' # Type of recurrent architecture to be integrated with U-Net

rnn_position = 'full' # Part of the U-Net with recurrent modules (encoder/decoder/full)

architecture = 'TimeDistributedUNet' # Architecture type

#architecture = 'UNetRNN'

normalization = 'instance' # Normalization type to apply in networks (None/batch/instance)

dropout = 0.5 # Dropout rate to apply in networks

base = 64 # Number of features to extract in architecture, in the first layer

kernel_size = 3 # Kernel size for convolutional architecture

padding = 1 # Padding for architecture

num_layers = 3 # Number of encoder and decoder layers to be used

#distance_layer = True # If True, use a layer with that computes distance maps as extra features at the end of the encoder 


if three_D or (not(three_D) and add3d > 0):
    
    if rnn is not None and rnn_position is not None:
        
        if supervision:
            
            network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_' + rnn + rnn_position + '_3DSupervision/'
                
        else:
    
            network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_' + rnn + rnn_position + '_3D/'
        
    elif rnn is not None and rnn_position is None:
        
        if supervision:
            
            network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_' + rnn + '_3DSupervision/'
            
        else:
        
            network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_' + rnn + '_3D/'
        
    elif rnn is None and rnn_position is None:
        
        if supervision:
            
            network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_3DSupervision/'
        
        else:
        
            network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_3D/'
    
else:
    
    if sum_work and not(supervision):

        network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_2Dextra/' # Folder where to save data related to Deep Learning architecture 
        
    elif sum_work and supervision:
        
        network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_2DextraSupervision/'
        
    elif not(sum_work) and supervision:
        
        network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_2DSupervision/'

    elif not(sum_work) and not(supervision):
        
        network_data_path = '/home/andres/Documents/_Data/Network_data_2/' + architecture + '_2D/' 
        

if k != 4:
    
    network_data_path = network_data_path.replace('Network_data_new','Network_data_' + str(k) + 'folds')
        

if autocontext:
    
    network_data_path = network_data_path[:-1] + str(autocontext_iter) + 'iter/'
        
# Training parameters

batch_size = 1

# How many inputs that will be loaded into RAM at a time.
# Recommended value around 128 but depends on the choice of augmentations and evalFrequency

RAM_batch_size = 1

# The maximum amount of samples to be uploaded to the GPU at the same time.
# Use conservative measure to prevent the program from crashing

batch_GPU_max = 32 # 16

batch_GPU_max_inference = 16 # 16

xav_init = 0 # Xavier initialization of network weights

# Class weights changes the importance of the different classes in the loss function
# By convention the weight should add up to 1
# length of list decides the class count

# Iterations to train for

I = 75000

# How often the model will be evaluated
# During testing (K=1) this is only used to show how far the network has come
    
eval_frequency = batch_size*5000

loss_frequency = batch_size*1000

opt = 'Adam' # Optimizer to be used. Can be Adam/RMSprop/SGD

class_weights = [0.2,0.8]

class_count = 2 # Classes used: foreground and background

lr = 0.00001 # Learning rate

lr_scheduling = False # Can be step or exponential

step = I/10 # Step for LR scheduling

lr_gamma = 0.5 # Decreasing factor for learning rate scheduling


# Loss function parameters

loss_fun = 'focal_supervision' # Loss function type. Can be dice, generalized_dice, focal, focal_cc, focal_dice, tversky, focal_tversky, exp_log, center, focal_center, focal_supervision, bce or bce_dice loss

loss_beta = 0.3

loss_gamma = 0.75

loss_gamma_exp_log = 0.3

loss_weights = [1,1]



metrics = ["Dice"]


#Do not change these

batch_GPU = min(batch_size, batch_GPU_max)


