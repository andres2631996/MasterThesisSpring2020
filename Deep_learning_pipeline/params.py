#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:11:02 2020

@author: andres
"""

import math

import numpy as np

# Data folders

excel_path = '/home/andres/Documents/_Data/CKD_Part1/' # Folder with file with Excel measurements on CKD1 flow measurements

excel_file = 'CKD_QFlow_results.xlsx' # Excel file with CKD1 measurements on flow

raw_path_ckd1 = '/home/andres/Documents/_Data/_Patients/_Raw/_ckd1/' # Folder with raw files from CKD1 study

# CKD flow measurements in CKD Part 2

raw_path_ckd2 = '/home/andres/Documents/_Data/CKD_Part2/4_Flow/' # Folder with flow files from CKD2 study

study2 = 'CKD2' # CKD2 study key

raw_path_hero = '/home/andres/Documents/_Data/Heroic/_Flow/' # Folder with flow files from Heroic study

studyh = 'Hero' # Heroic study key

raw_path_extr = '/home/andres/Documents/_Data/Extra/_Flow/' # Folder with flow files from Extra study

studye = 'Extr' # Extra study key



flow_folders = [raw_path_ckd2, raw_path_hero, raw_path_extr] # Folders with flow information for patient stratification

studies_flow = ['CKD2', 'Hero', 'Extr'] # Studies with flow information



rep = True # Repetition of minority studies in data presented to the network (oversampling: CKD2*1, Heroic*3, Extra*5)


# Data selection parameters

prep_step = 'crop' # Level of preprocessing to apply for images to the network (raw/prep/crop)

train_with = 'magBF' # Type of images to train with (mag_: magnitude, magBF: magnitude corrected in artifacts, pha: phase, both: magnitude +phase, bothBF: magnitude corrected in artifacts + phase). Can include "oth" to train also with the third existing modality in Philips and Siemens images, and "othmip", to train with the MIPs of those images

three_D = False # Train with separate 2D slices or 2D + time volumes

add3d = 2 # Number of past and future neighboring slices to build a 2.5D dataset (Set 0 for 2D)

sum_work = False # If True, include an extra channel with the sum and MIP of all frames along time, else dont (only in 2D) (set False for 2D+time)

channel_count = 1 # Channel counter (mostly unused)

multi_view = False # If True, allow for a multi-view analysis along time (normally is always False)

supervision = True # If True, perform deep supervision on given architecture

test = False # State if model is being trained (False) or tested/validated (True)

autocontext = False # Cascaded network if True, otherwise False

autocontext_iter = 2 # Autocontext or cascading iterations (autocontext must be True)

# Augmentation parameters

augmentation = True # If True, allow for augmentation during traing

augm_params = [0.15, 10] # 3D Augmentation limit parameters: maximum zoom increase, maximum rotation in degrees

augm_probs = [0.5]*5 # Probabilities for each augmentation event

# 2D augmentation with albumentations

augm2D_limits = [0.95, 10] # Augmentation limit parameters: Scale, Rotation, Flip

augm2D_probs = [0.5]*5 # Probabilities for each augmentation event (the same as augm_probs)

# Cross-validation parameters

k = 4 # Number of cross validation folds

# Architecture parameters

rnn = 'GRU' # Type of recurrent architecture to be integrated with U-Net (GRU/LSTM) (None for 2D)

rnn_position = 'full' # Part of the U-Net with recurrent modules (encoder/decoder/full) (None for 2D)

architecture = 'TimeDistributedUNet' # Architecture type (see crossValidation.py or test.py to see all the possible types)

normalization = 'instance' # Normalization type to apply in networks (None/batch/instance)

dropout = 0.5 # Dropout rate to apply in networks (if dropout is desired to be applied)

base = 64 # Number of features to extract in architecture, in the first layer

kernel_size = 3 # Kernel size for convolutional architecture

padding = 1 # Padding for architecture

num_layers = 3 # Number of encoder and decoder layers to be used


# Decide on name of folder where to save trained models and training and validation results (FOLDER MUST BE CREATED BEFORE, OTHERWISE CODE DOES NOT WORK)

if three_D or (not(three_D) and add3d > 0):
    
    if rnn is not None and rnn_position is not None:
        
        if supervision:
            
            network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_' + rnn + rnn_position + '_3DSupervision/'
                
        else:
    
            network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_' + rnn + rnn_position + '_3D/'
        
    elif rnn is not None and rnn_position is None:
        
        if supervision:
            
            network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_' + rnn + '_3DSupervision/'
            
        else:
        
            network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_' + rnn + '_3D/'
        
    elif rnn is None and rnn_position is None:
        
        if supervision:
            
            network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_3DSupervision/'
        
        else:
        
            network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_3D/'
    
else:
    
    if sum_work and not(supervision):

        network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_2Dextra/' # Folder where to save data related to Deep Learning architecture 
        
    elif sum_work and supervision:
        
        network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_2DextraSupervision/'
        
    elif not(sum_work) and supervision:
        
        network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_2DSupervision/'

    elif not(sum_work) and not(supervision):
        
        network_data_path = '/home/andres/Documents/_Data/Network_data_final/' + architecture + '_2D/' 

# If use a different number of folds than 4 or use cascading, change name of final folder (folder must be previously existing)

if k != 4:
    
    network_data_path = network_data_path.replace('Network_data_final','Network_data_final_' + str(k) + 'folds')
        

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
    
eval_frequency = batch_size*1000 # How often to do validation and print validation results

loss_frequency = batch_size*1000 # How often to print the loss function value (may be different than eval_frequency, cannot be 1)

opt = 'Adam' # Optimizer to be used. Can be Adam/RMSprop/SGD

class_weights = [0.2,0.8] # Class weighting for loss function (mostly unused)

class_count = 2 # Classes used: foreground and background

lr = 0.000001 # Learning rate

lr_scheduling = False # Can be step or exponential

step = I/10 # Step for LR scheduling

lr_gamma = 0.5 # Decreasing factor for learning rate scheduling


# Loss function parameters

loss_fun = 'focal_supervision' # Loss function type. Can be dice, generalized_dice, focal, focal_cc, focal_dice, tversky, focal_tversky, exp_log, center, focal_center, focal_supervision, bce or bce_dice loss

# If deep supervision is applied, the name of the function includes "supervision", as it is a special loss taking into account the output 

# Parameters for Tversky and Focal-Tversky loss

loss_beta = 0.3

loss_gamma = 0.75

loss_gamma_exp_log = 0.3

loss_weights = [1,1]



metrics = ["Dice"]


#Do not change these

batch_GPU = min(batch_size, batch_GPU_max)


