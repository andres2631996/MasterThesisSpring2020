#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:56:09 2020

@author: andres
"""

import sys

import os

import time

import datetime

import shutil

import copy

import torch

import torch.nn as nn

import numpy as np

import random

import SimpleITK as sitk

import matplotlib.pyplot as plt

from datasets import QFlowDataset

from patientStratification import StratKFold


def preprocessingType(preprocessing):
        
    """
    Provide parent folder where to extract data depending on the type of 
    preprocessing desired.
    
    Params:
        
        - inherited from class (check at the beginning of the class)
    
    Returns:
        
        - start_folder: parent folder
    
    """
    
    if preprocessing == 'raw' or preprocessing == 'Raw' or preprocessing == 'RAW':

        start_folder = '/home/andres/Documents/_Data/_Patients/_Raw/'

    elif preprocessing == 'prep' or preprocessing == 'Prep' or preprocessing == 'PREP':
        
        start_folder = '/home/andres/Documents/_Data/_Patients/_Prep/'
    
    elif preprocessing == 'crop' or preprocessing == 'Crop' or preprocessing == 'CROP':
        
        start_folder = '/home/andres/Documents/_Data/_Patients/_Pre_crop/'
    
    else:
        
        print('\nWrong pre-processing step introduced. Please introduce a valid pre-processing step\n')
    
    return start_folder




#def get_datasets(paths_img, train, work_with, threeD):
#    
#    
#    """
#    Creates a list of Dataset objects. A Dataset object contains all paths to a unique patients slices
#    
#    :param paths_img: where the input data is located
#    
#           train: whether dataset is for training or for validation
#    
#    :returns: list of datasets with paths to every patient slice
#    """
#
#    datasets = []
#
#
#    #Create a dataset for current patient
#    
#    datasets.append(QFlowDataset(paths_img, train, work_with, threeD))
#
#    #Pseudo randomness is needed to compare different sets of hyperparameters fairly
#    
#    random.seed(0)
#    random.shuffle(datasets)
#
#    return datasets


#def get_data_loader(datasets, k, K, data_path, datasets_eval=None):
#    
#    
#    """
#    Creates one DataLoader for validation and one for training from a list of datasets. 
#    
#    Writes the patient partioning into training and validation into a text file.
#    
#    :param datasets: List of datasets where each datset contain paths to a unique patients slices
#    
#    :param k: Current K-fold iteration
#    
#    :param K: Max K-fold iteration
#    
#    :param data_path: Path to output folder
#    
#    :return: loader_train, loader_val
#    """
#
#    if K == 1:
#        
#        if datasets_eval:
#            
#            train_set = torch.utils.data.ConcatDataset(datasets)
#            
#            eval_set = torch.utils.data.ConcatDataset(datasets_eval)
#            for i in range(len(eval_set.datasets)):
#                eval_set.datasets[i].setTrain(0)
#                if params.imiomics:
#                    eval_set.datasets[i].set_atlas_path(params.imiomics, k, "V")
#
#            loader_val = torch.utils.data.DataLoader(eval_set,
#                                            num_workers=0, # Slave Processes for fetching data
#                                            batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time
#                                            pin_memory=True) # Pins memory, enables nonblocking communication with GPU
#        else:
#            print("Creating final network. No validation.")
#            train_set = torch.utils.data.ConcatDataset(datasets)
#            loader_val = None
#    else:
#        num_total = len(datasets)
#        num_eval = int(num_total/K)
#
#        if num_eval < 1:
#            print("ERROR: Some evaluation sets will be empty")
#            exit()
#        else:
#            print("Cross-validation. Each validation set will contain " + str(num_eval) + " patient series")
#
#        # Creates a dataset for validation of a previously scrambled list of datasets
#        if k == K-1:
#            indices = range(int(k*num_eval), int(num_total))
#        else:
#            indices = range(int(k*num_eval), int((k+1)*num_eval))
#        eval_sets = torch.utils.data.Subset(datasets, indices)
#        eval_set = torch.utils.data.ConcatDataset(eval_sets)
#
#        #Set training to false, no augmentation etc
#        for i in range(len(eval_set.datasets)):
#            eval_set.datasets[i].setTrain(0)
#            if params.imiomics:
#                eval_set.datasets[i].set_atlas_path(params.imiomics, k, "V")
#
#        loader_val = torch.utils.data.DataLoader(eval_set,
#                                            num_workers=0, # Slave Processes for fetching data
#                                            batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time
#                                            pin_memory=True) # Pins memory, enables nonblocking communication with GPU
#
#        # Creates a dataset for training of those datasets that are left
#        if k == K-1:
#            indices = np.delete(np.arange(num_total), range(int(k*num_eval), int(num_total)))
#        else:
#            indices = np.delete(np.arange(num_total), range(int(k*num_eval), int((k+1)*num_eval)))
#            
#        train_sets = torch.utils.data.Subset(datasets, indices)
#        train_set = torch.utils.data.ConcatDataset(train_sets)
#
#        #Write patientPartioning files
#        with open(data_path + "PatientPartioning_k=" + str(k), 'w') as file:
#            file.write('Patients used for training')
#            file.write('\n')
#            patientIDs = []
#            for dataset in train_sets:
#                patientIDs.append(dataset.img_paths[0].split("/")[-1].split('_')[0])
#            patientIDs.sort()
#            for patientID in patientIDs:
#                file.write(patientID)
#                file.write('\n')
#
#            file.write('Patients used for validation')
#            file.write('\n')
#            patientIDs = []
#            for dataset in eval_sets:
#                patientIDs.append(dataset.img_paths[0].split("/")[-1].split('_')[0])
#            patientIDs.sort()
#            for patientID in patientIDs:
#                file.write(patientID)
#                file.write('\n')
#    
#    #Sets training to true, augmentations etc
#    for i in range(len(train_set.datasets)):
#        train_set.datasets[i].setTrain(1)
#        if params.imiomics:
#            train_set.datasets[i].set_atlas_path(params.imiomics, k, "T")   
#
#    #Finds weights for the weighted sampler
#    weights = get_sample_weights(train_set, params.sample_frequncy_no_lesions)            
#    weights = torch.DoubleTensor(weights)                                       
#    lesion_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, params.RAM_batch_size, replacement=True)                     
#
#    loader_train = torch.utils.data.DataLoader(train_set,
#                                        num_workers=0, #4 Slave Processes for fetching data
#                                        batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time
#                                        #shuffle=True # Re-shuffles the data at every epoch 
#                                        sampler=lesion_sampler, #Sampler that prioritizes samples with lesions
#                                        pin_memory=True) # Pins memory, enables nonblocking communication with GPU
#
#    if K==1 and params.patients_to_be_moved:
#        moved_patients_img, moved_patients_seg = loader_val.remove_paths(params.patient_to_be_moved)
#        loader_train.add_paths(moved_patients_img, moved_patients_seg)
#        print("Moved patients:", moved_patients_img, moved_patients_seg)
#
#    return loader_train, loader_val


# Code for data loading

# Flags
    
# Pre-processing step from where to reach data: 'raw', 'prep' or 'crop'
    
prep_step = 'raw' 

t1 = time.time()


# Decide whether to train with magnitude, with phase images or with both

train_with = 'both'


# Get list with stratified patient folders and information on test data
    
rep = True # Factor allowing for repetition of patients from minority studies (Heroic and Extra)

# Import information of EXCEL file with flow measurements of CKD1

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

k = 5

train = True

threeD = True

augmentation = True
    

strKFolds = StratKFold([raw_path_ckd2, raw_path_hero, raw_path_extr], excel_path, 
                            excel_file, raw_path_ckd1,[study2, studyh, studye], k, rep)

patient_paths, test_img, test_flow = strKFolds.__main__()

cont = 0

for fold in patient_paths:
    
    patient_paths[cont] = [preprocessingType(prep_step) + path for path in fold]

    cont += 1

probs = [1, 1, 1, 1]

dataset = QFlowDataset(patient_paths[1], train, train_with, threeD, augmentation, probs)

test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = 0, shuffle=True)

for i,images in enumerate(test_loader):
    
    if i == 0:
        
        break

test_images = images # Test images

t2 = time.time()

print('Elapsed time: {} seconds'.format(t2 - t1))

x = np.squeeze(test_images[0].numpy())

y = np.squeeze(test_images[1].numpy())

center = y.shape[-1]//2

plt.figure(figsize = (13,5))

plt.subplot(131)

plt.imshow(x[0,:,:,0,0], cmap = 'gray'), plt.colorbar()

plt.subplot(132)

plt.imshow(x[0,:,:,0,1], cmap = 'gray'), plt.colorbar()

plt.subplot(133)

plt.imshow(y[0,:,:,0], cmap = 'gray'), plt.colorbar()
