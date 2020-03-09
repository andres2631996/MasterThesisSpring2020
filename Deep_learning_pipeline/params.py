#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:34:14 2020

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


# Data selection parameters

prep_step = 'raw' # Level of preprocessing to apply for images to the network

train_with = 'mag_' # Type of images to train with

three_D = False # Train with separate 2D slices or 2D + time volumes

channel_count = 1 # Channel counter


# Augmentation parameters

augmentation = True

augm_params = [0.15, 0.5, 30] # Augmentation limit parameters: maximum mean noise amplitude, maximum scale and maximum degree rotation

augm_probs = [0.5]*5 # Probabilities for each augmentation event

# Optional 2D augmentation with albumentations

augm2D_limits = [0.5, 45] # Augmentation limit parameters: contrast, Brightness, Scale, Rotation, Flip, Elastic Deformation

augm2D_probs = [0.5]*3

# Cross-validation parameters

k = 4 # Number of cross validation folds

rep = True # Repetition of minority studies in data presented to the network



# Architecture parameters

architecture = 'UNet_with_Residuals' # Architecture type

normalization = 'instance' # Normalization type to apply in networks (None/batch/instance)

dropout = 0.0 # Dropout rate to apply in networks

layers = 4 # Number of architecture layers

base = 64 # Basic number of features to extract in architecture

kernel_size = 3 # Kernel size for convolutional architecture

padding = 1 # Padding for architecture

network_data_path = '/home/andres/Documents/_Data/Network_data/' + architecture + '/' # Folder where to save data related to Deep Learning architecture





# Training parameters

batch_size = 1

# How many inputs that will be loaded into RAM at a time.
# Recommended value around 128 but depends on the choice of augmentations and evalFrequency

RAM_batch_size = 128

# The maximum amount of samples to be uploaded to the GPU at the same time.
# Use conservative measure to prevent the program from crashing

batch_GPU_max = 16 # 16

batch_GPU_max_inference = 16 # 16

xav_init = 0 # Xavier initialization of network weights

# Class weights changes the importance of the different classes in the loss function
# By convention the weight should add up to 1
# length of list decides the class count

# Iterations to train for

I = 10000

# How often the model will be evaluated
# During testing (K=1) this is only used to show how far the network has come
    
eval_frequency = batch_size*100

opt = 'Adam' # Optimizer to be used. Can be Adam/RMSprop/SGD

class_weights = [0.2,0.8]

class_count = 2 # Classes used: foreground and background

lr = 0.0001 # Learning rate

lr_scheduling = False # Can be step or exponential

step = I/10 # Step for LR scheduling

lr_gamma = 0.5 # Decreasing factor for learning rate scheduling





# Loss function parameters

loss_fun = 'focal_tversky' # Loss function type. Can be dice, generalized_dice, focal, focal_tversky, exp_log, bce or bce_dice loss

loss_beta = 0.3

loss_gamma = 0.75

loss_gamma_exp_log = 0.3

loss_weights = [1,1]



metrics = ["Dice", "Precision", "Recall"]


#Do not change these

batch_GPU = min(batch_size, batch_GPU_max)

checkpoint_loading = False # Tells if we are training from some checkpoint
