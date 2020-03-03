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

import torch.optim as optim

import numpy as np

import random

import SimpleITK as sitk

import matplotlib.pyplot as plt

from datasets import QFlowDataset

from patientStratification import StratKFold

import params

import itertools

import utilities

import train


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





def get_data_loader(datasets, k, K, data_path, datasets_eval=None):
    
    
    """
    Creates one DataLoader for validation and one for training from a list of datasets. 
    
    Writes the patient partioning into training and validation into a text file.
    
    :param datasets: List of lists of folders with image paths with datasets for training
    
    :param k: Current K-fold iteration
    
    :param K: Max K-fold iteration
    
    :param data_path: Path to output folder
    
    :param datasets_eval: List of folders with image paths with datasets for validation
    
    :return: loader_train, loader_val
    """

    
    if len(datasets) ==  0:
        
        print('\nNo available folders for data training. Please introduce valid folders for data training\n')
        
        exit()
    
    else:
    
        train_dataset = QFlowDataset(train_paths, train = True, params.train_with,
                                     params.three_D, params.augmentation, params.probs)
        
    
        if K == 1:
            
            if datasets_eval:
                
                train_set = torch.utils.data.ConcatDataset(datasets)
                
                eval_set = torch.utils.data.ConcatDataset(datasets_eval)
                
                for i in range(len(eval_set.datasets)):
                    
                    eval_set.datasets[i].setTrain(0)
                    
    
                loader_val = torch.utils.data.DataLoader(eval_set,
                                                         num_workers=0, # Slave Processes for fetching data
                                                         batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time
                                                         pin_memory=True) # Pins memory, enables nonblocking communication with GPU
            
            else:
                
                print("Creating final network. No validation.")
                
                train_dataset = QFlowDataset(datasets, train = True, test = False, params.train_with,
                                             params.three_D, params.augmentation, params.probs)
                    
                train_dataset.setTrain(1)
                
                loader_train = torch.utils.data.DataLoader(train_dataset,
                                                           num_workers=0, #4 Slave Processes for fetching data
                                                           batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time
                                                           pin_memory=True) # Pins memory, enables nonblocking communication with GPU
    
                
                loader_val = None
                
                
        else:
            
            num_total = len(datasets)
            
            num_eval = int(num_total/K)
    
            if num_eval < 1:
                
                print("ERROR: Some evaluation sets will be empty")
                
                exit()
                
            else:
                
                print("Cross-validation. Each validation set will contain " + str(num_eval) + " patient series")
                
                
                if datasets_eval is None:
                    
                    print('\nNo available folders for data validation. Please introduce valid folders for data validation\n')
                    
                    exit()
                
                else:
                    
                    val_dataset = QFlowDataset(datasets_eval, train = False, test = False, 
                                               params.train_with, params.three_D, params.augmentation, 
                                               params.probs)
                    
                    val_dataset.setTrain(0)
                    
                    train_dataset = QFlowDataset(datasets, train = True, test = False, 
                                                 params.train_with,params.three_D, 
                                                 params.augmentation, params.probs)
                    
                    train_dataset.setTrain(1)
                
    
                    loader_val = torch.utils.data.DataLoader(val_dataset,
                                                             num_workers=0, # Slave Processes for fetching data
                                                             batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time
                                                             shuffle=True,
                                                             pin_memory=True) # Pins memory, enables nonblocking communication with GPU
                    
                    loader_train = torch.utils.data.DataLoader(train_dataset,
                                                               num_workers=0, #4 Slave Processes for fetching data
                                                               batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time
                                                               shuffle=True, # Re-shuffles the data at every epoch
                                                               pin_memory=True) # Pins memory, enables nonblocking communication with GPU
    
            
    
            #Write patientPartioning files
            
            with open(data_path + "PatientPartioning_k=" + str(k), 'w') as file:
                
                file.write('Patients used for training')
                
                file.write('\n')
                
                patientIDs = []
                
                for train_path in train_paths:
                    
                    patientIDs.append(train_path.split('/')[-1])
                    
                patientIDs.sort()
                
                for patientID in patientIDs:
                    
                    file.write(patientID)
                    file.write('\n')
    
                file.write('Patients used for validation')
                
                file.write('\n')
                
                patientIDs = []
                
                for val_path in datasets_eval:
                    
                    patientIDs.append(val_path.split('/')[-1])
                    
                patientIDs.sort()
                
                for patientID in patientIDs:
                    
                    file.write(patientID)
                    
                    file.write('\n')
        
    
#        if K==1 and params.patients_to_be_moved:
#            moved_patients_img, moved_patients_seg = loader_val.remove_paths(params.patient_to_be_moved)
#            loader_train.add_paths(moved_patients_img, moved_patients_seg)
#            print("Moved patients:", moved_patients_img, moved_patients_seg)
    
        return loader_train, loader_val


# Get list of lists with stratified patients according to their flow 

strKFolds = StratKFold(params.flow_folders, params.excel_path, 
                            params.excel_file, params.raw_path_ckd1, params.studies_flow, params.k, params.rep)

patient_paths, test_img, test_flow = strKFolds.__main__()

data_path = params.network_data_path #output folder path

K = params.k #K in K-fold cross validation

cont = 0

for fold in patient_paths:
    
    patient_paths[cont] = [preprocessingType(params.prep_step) + path for path in fold]

    cont += 1


# Copies files to output path to help remembering the setup
    
shutil.copyfile("params.py", data_path + "params.py")

shutil.copyfile("crossValidation.py", data_path + "crossValidation.py")

shutil.copyfile("datasets.py", data_path + "datasets.py")

shutil.copyfile("train.py", data_path + "train.py")

shutil.copyfile("evaluate.py", data_path + "evaluate.py")

if params.RAM_batch_size%params.batch_size != 0:
    
    print("ERROR: RAM_batch_size must be evenly dividable with batch_size")
    
    exit()
    
# Data time

starting_time = time.time()

curr_time = datetime.datetime.now().time()

curr_date = datetime.datetime.now().date()

losses = []

metrics_train_array = np.zeros((params.k, 2, len(params.metrics)))

metrics_val_array = np.zeros((params.k, 2, len(params.metrics)))

    

for k in range(K):
    
    utilities.clear_GPU()
    
    # Initialize network
    
    print("Initializing network, fold " + str(k) + "...")
    
    if params.archictecture == "UNet_with_Residuals":
        
        pass
        
        #net = UNetResiduals(channel_count=no_channels, class_count=params.class_count).cuda() FIX LATER!!
        
#        
#    elif params.arch_type == "VGG11":
#        net = UNet11(channel_count=no_channels, pre_trained=True).cuda()
    else:
        
        print("Error: Architecture " + params.architecture + " not found.")
    
    
    # Xavier initialization of CNN weights
    
    if params.xav_init == 1:
        
        net.apply(weights_init)
    
    
    val_paths = patient_paths[k] # List with patient paths for validation
    
    
    if k == 0: # First fold
    
        train_paths = patient_paths[1:]
        
        train_paths = list(itertools.chain.from_iterable(train_paths))
    
    elif k == K - 1: # Last fold
        
        train_paths = patient_paths[:-1]
        
        train_paths = list(itertools.chain.from_iterable(train_paths))
    
    else: # Intermediate folds
        
        train_paths = patient_paths[:k] + patient_paths[(k+1):] # List with patient paths for training
        
            
    # Training and validation data loaders

    loader_train, loader_val = get_data_loader(train_paths, k, K, data_path, val_paths) 
    
    # Train and evaluate
    
    print("\nStart training network, fold " + str(k) + "...\n")

    loss, metrics_train, metrics_val, model_state, optimizer, optimizer_state = train.train(net, loader_train, loader_val, 
                                                                                            params.k, params.eval_frequency, 
                                                                                            params.I)
    
    
    # Save model and optimizer for later inference
    
    utilities.model_saving(model_state, optimizer_state, params.network_data_path, 'FinalTrainedWith' + params.train_with + '_' + params.prep_step + 'fold_' + str(params.k) + '.tar')
    
    losses.append(loss)
    
    metrics_array = np.array(metrics_train)
    
    metrics_names = np.unique(metrics_array[:,0])
    
    for i in range(len(metrics_train)):
        
        # Store arrays with mean and standard deviation results
        
        metrics_train_array[k,:,i] = metrics_train[i][1:]
        
        metrics_val_array[k,:,i] = metrics_val[i][1:]



# Print cross-validation results and show figures
        
mean_loss = np.mean(np.array(losses))

std_loss = np.std(np.array(losses))

print('\nFinal {} loss over folds: {} +- {}\n'.format(params.loss_fun, mean_loss, std_loss))

fig = plt.figure(figsize = (13,5))

plt.bar(np.arange(1,K),losses, color = 'b')

plt.title(params.loss_fun + ' over folds')

plt.xlabel('Fold number')

plt.ylabel(params.loss_fun + ' loss')

fig.savefig(params.network_data_path + 'loss_plot_' + str(K) + '_folds_' + 'trainedWith' + params.train_with + '_' + params.prep_step + '.png')



    
for i in range(len(metrics_train)):
    
    mean_train = np.mean(metrics_train_array[:,0,i].flatten())

    std_train = np.mean(metrics_train_array[:,1,i].flatten())
    
    mean_val = np.mean(metrics_val_array[:,0,i].flatten())

    std_val = np.mean(metrics_val_array[:,1,i].flatten())
    
    print('Final {} over training folds: {} +- {}\n'.format(metrics_names[i], mean_train, std_train))
    
    print('Final {} over validation folds: {} +- {}\n'.format(metrics_names[i], mean_val, std_val))
    
    # Show in figures cross-validation results, too
    
    # Results for training
       
    fig = plt.figure(figsize = (13,5))

    plt.bar(np.arange(1,K), metrics_train_array[:,0,i], color = 'b', yerr = metrics_train_array[:,1,i])
    
    plt.title(metrics_names[i] + ' over training folds')
    
    plt.xlabel('Fold number')
    
    plt.ylabel(metrics_names[i])
    
    fig.savefig(params.network_data_path + metrics_names[i] + '_training_plot_' + str(K) + '_folds_' + 'trainedWith' + params.train_with + '_' + params.prep_step + '.png')
    
    
    # Results for validation
    
    fig = plt.figure(figsize = (13,5))

    plt.bar(np.arange(1,K), metrics_val_array[:,0,i], color = 'r', yerr = metrics_val_array[:,1,i])
    
    plt.title(metrics_names[i] + ' over validation folds')
    
    plt.xlabel('Fold number')
    
    plt.ylabel(metrics_names[i])
    
    fig.savefig(params.network_data_path + metrics_names[i] + '_validation_plot_' + str(K) + '_folds_' + 'trainedWith' + params.train_with + '_' + params.prep_step + '.png')
    

    
    
#dataset = QFlowDataset(patient_paths[-2], params.train, params.train_with, params.threeD, params.augmentation)

#test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = 0, shuffle=True)
#
#for i,images in enumerate(test_loader):
#    
#    if i == 0:
#        
#        break
#
#test_images = images # Test images
#
#t2 = time.time()
#
#print(t2-t1)
#
#x = np.squeeze(test_images[0].numpy())
#
#y = np.squeeze(test_images[1].numpy())
#
#center = y.shape[-1]//2
#
#plt.figure(figsize = (13,5))
#
#plt.subplot(141)
#
#plt.imshow(x[0,:,:,20,0], cmap = 'gray'), plt.colorbar()
#
#plt.subplot(142)
#
#plt.imshow(x[0,:,:,20,1], cmap = 'gray'), plt.colorbar()
#
#plt.subplot(143)
#
#plt.imshow(y[0,:,:,20], cmap = 'gray'), plt.colorbar()
#
#plt.subplot(144)
#
#plt.imshow(x[0,:,:,20,0]*y[0,:,:,20], cmap = 'gray'), plt.colorbar()
