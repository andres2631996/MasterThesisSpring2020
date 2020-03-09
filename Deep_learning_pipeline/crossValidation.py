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

import models


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


def weights_init(m):
    
    """
    Initializes the weights in the layers with Xavier intialization
    :param m: The model in question
    :return:
    """
    
    if isinstance(m, torch.nn.Conv2d):
        
        torch.nn.init.xavier_uniform_(m.weight.data)
        
        if m.bias is not None:
            
            torch.nn.init.zeros_(m.bias)


def extractVTKfilesStratification(patient_paths):
    
    """
    From a list of lists with stratified patient paths, generate a stratified
    list of lists with the correspondent VTK files
    
    Params:
        
        - patient_paths. list of lists with stratified patient paths
    
    
    Returns:
        
        - raw_paths and mask_paths (paths of files to use)
    
    """

    cont = 0


    if 'both' in params.train_with:
        
        mag_paths = []
        
        mag_path = []
        
        pha_paths = []
        
        pha_path = []
    
    else:
    
        raw_paths = []
        
        raw_path = []
    
    
    mask_paths = []
    
    mask_path = []
    
    for fold in patient_paths:
        
        patient_paths[cont] = [preprocessingType(params.prep_step) + path for path in fold]
    
        for patient_path in patient_paths[cont]: # Look for all images in the given paths
        
            images = sorted(os.listdir(patient_path)) 
        
            if 'both' in params.train_with:
                
                if params.train_with == 'bothBF':
                    
                    ind_mag = [i for i,s in enumerate(images) if 'magBF' in s]
                
                elif params.train_with == 'both':
                    
                    ind_mag = [i for i,s in enumerate(images) if 'mag_' in s]
    
                ind_pha = [i for i,s in enumerate(images) if 'pha' in s]
                
                ind_msk = [i for i,s in enumerate(images) if 'msk' in s]
                
                for ind_m, ind_p, ind_mk in zip(ind_mag, ind_pha, ind_msk):
                    
                    mag_path.append(patient_path + images[ind_m])
                    
                    pha_path.append(patient_path + images[ind_p])
                    
                    mask_path.append(patient_path + images[ind_mk])
                
            
            else:
                
                ind_raw = [i for i,s in enumerate(images) if params.train_with in s]
                
                ind_msk = [i for i,s in enumerate(images) if 'msk' in s]
                
                for ind, ind_m in zip(ind_raw,ind_msk):
                    
                    raw_path.append(patient_path + images[ind])
                    
                    mask_path.append(patient_path + images[ind_m])
                    
                    
                    
                    
        if 'both' in params.train_with:
            
            mag_paths.append(mag_path)
            
            pha_paths.append(pha_path)
            
            mask_paths.append(mask_path)
            
            mag_path = []
            
            pha_path = []
            
            mask_path = []
    
        else:
            
            raw_paths.append(raw_path)
                
            mask_paths.append(mask_path)
            
            raw_path = []
        
            mask_path = []
    
        cont += 1
    
    if 'both' in params.train_with:
        
        raw_paths = np.array([mag_paths, pha_paths])
        
        return raw_paths, mask_paths
    
    else:
        
        raw_paths = np.expand_dims(np.array(raw_paths),0)
        
        return raw_paths, mask_paths
    
    
def pathExtractorCrossValidation(k, m_paths, r_paths, patient_paths):

    """
    Extract raw files and mask files for each cross validation fold.
    
    Params:
        
        k: fold index of cross validation
        
        m_path: path with stratified mask files
        
        r_path: path with stratified raw files
        
        patient_paths: path with stratified patients
        
    
    Returns:
        
        train_raw_path, train_mask_path, val_raw_path, val_mask_path, train_patients, val_patients


    """    
    
    val_raw_paths = r_paths[:,k].tolist() # List with patient paths for validation
    
    val_mask_paths = m_paths[k]
    
    val_patients = patient_paths[k]
    
    if k == 0: # First fold 
        
        train_raw_paths = r_paths[:,1:].tolist()      #r_paths[:,0].tolist(), m_paths[0]
    
        train_mask_paths = m_paths[1:]
        
        train_mask_paths = list(itertools.chain.from_iterable(train_mask_paths))
        
        train_patients = patient_paths[1:]
        
        train_patients = list(itertools.chain.from_iterable(train_patients))
        
        train_raw_paths_both = []
        
        for i in range(len(train_raw_paths)):
            
            train_raw_paths_both.append(list(itertools.chain.from_iterable(train_raw_paths[i])))
    
    elif k == K - 1: # Last fold
        
        train_raw_paths = r_paths[:,:-1].tolist()      #r_paths[:,0].tolist(), m_paths[0]
    
        train_mask_paths = m_paths[:-1]
        
        train_mask_paths = list(itertools.chain.from_iterable(train_mask_paths))
        
        train_patients = patient_paths[:-1]
        
        train_patients = list(itertools.chain.from_iterable(train_patients))
        
        train_raw_paths_both = []
        
        for i in range(len(train_raw_paths)):
            
            train_raw_paths_both.append(list(itertools.chain.from_iterable(train_raw_paths[i])))
    
    else: # Intermediate folds
        
        train_raw_paths_before = r_paths[:,:k].tolist()      #r_paths[:,0].tolist(), m_paths[0]
        
        train_patients = list(itertools.chain.from_iterable(patient_paths[:k] + patient_paths[(k + 1):]))
        
        train_raw_paths_both_before = []
        
        for i in range(len(train_raw_paths_before)):
            
            train_raw_paths_both_before.append(list(itertools.chain.from_iterable(train_raw_paths_before[i])))
            
        
        train_raw_paths_after = r_paths[:,(k + 1):].tolist()      #r_paths[:,0].tolist(), m_paths[0]
        
        train_raw_paths_both_after = []
        
        for i in range(len(train_raw_paths_after)):
            
            train_raw_paths_both_after.append(list(itertools.chain.from_iterable(train_raw_paths_after[i])))
            
        
        
        train_raw_paths_both = []
        
        for i in range(len(train_raw_paths_before)):
            
            train_raw_paths_both.append(list(itertools.chain.from_iterable(train_raw_paths_before[i] + train_raw_paths_after[i])))
        
        
        train_mask_paths = m_paths[:k] + m_paths[(k+1):] # List with patient paths for training
        
        train_mask_paths = list(itertools.chain.from_iterable(train_mask_paths))
        
    
    return train_raw_paths_both, train_mask_paths, val_raw_paths, val_mask_paths, train_patients, val_patients
    
    

def get_data_loader(train_raw_paths, train_mask_paths, k, K, data_path, val_raw_paths, val_mask_paths, train_patients, val_patients):
    
    
    """
    Creates one DataLoader for validation and one for training from a list of datasets. 
    
    Writes the patient partitioning into training and validation into a text file.
    
    :param train_raw_paths: List of lists of folders with raw files paths for training
    
    :param train_mask_paths: List of lists of folders with mask files paths for training
    
    :param k: Current K-fold iteration
    
    :param K: Max K-fold iteration
    
    :param data_path: Path to output folder
    
    :param val_raw_paths: List of lists of folders with raw files paths for validation
    
    :param val_mask_paths: List of lists of folders with mask files paths for validation
    
    :param train_patients: list of lists with stratified patients for training
    
    :param val_patients: list of lists with stratified patients for validation
    
    :return: loader_train, loader_val
    """

    
    if len(train_raw_paths) ==  0:
        
        print('\nNo available folders for data training. Please introduce valid folders for data training\n')
        
        exit()
    
    else:
        
        train = True
        
        test = False 
    
        train_dataset = QFlowDataset(train_raw_paths, train_mask_paths, train, params.train_with,
                                     params.three_D, params.augmentation)
        
        
        
        if K == 1:
            
                
            print("Creating final network. No validation.")
            
            loader_train = torch.utils.data.DataLoader(train_dataset,
                                                       num_workers=0, #4 Slave Processes for fetching data
                                                       batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time
                                                       pin_memory=True) # Pins memory, enables nonblocking communication with GPU

            
            loader_val = None
                
                
        else:
            
            num_total = len(train_mask_paths)
            
            num_eval = int(num_total/K)
    
            if num_eval < 1:
                
                print("ERROR: Some evaluation sets will be empty")
                
                exit()
                
            else:
                
                print("Cross-validation. Each validation set will contain " + str(num_eval) + " files")

                    
                train = False
                
                test = False
                
                val_dataset = QFlowDataset(val_raw_paths, val_mask_paths, train, 
                                           params.train_with, params.three_D, params.augmentation)


                loader_val = torch.utils.data.DataLoader(val_dataset,
                                                         num_workers=0, # Slave Processes for fetching data
                                                         batch_size=params.RAM_batch_size, #Number of samples to be loaded into RAM at a time,
                                                         pin_memory=True) # Pins memory, enables nonblocking communication with GPU
                
                loader_train = torch.utils.data.DataLoader(train_dataset,
                                                           num_workers=0, #4 Slave Processes for fetching data
                                                           batch_size=params.batch_size, #Number of samples to be loaded into RAM at a time 
                                                           pin_memory=True) # Pins memory, enables nonblocking communication with GPU
    
            
    
            #Write patientPartioning files
            
            with open(data_path + "PatientPartioning_k=" + str(k), 'w') as file:
                
                file.write('Patients used for training')
                
                file.write('\n')
                
                patientIDs = []
                
                for train_patient in train_patients:
                    
                    patientIDs.append(train_patient.split('/')[-2])
                        
                patientIDs.sort()
                
                for patientID in patientIDs:
                    
                    file.write(patientID)
                    file.write('\n')
    
                file.write('Patients used for validation')
                
                file.write('\n')
                
                patientIDs = []
                
                for val_patient in val_patients:
                    
                    patientIDs.append(val_patient.split('/')[-2])
                    
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

r_paths, m_paths = extractVTKfilesStratification(patient_paths)

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
    
    if params.architecture == "UNet_with_Residuals":
        
        net = models.UNet_with_Residuals(2).cuda()
        
        print(net)
        
        utilities.print_num_params(net)
#        
#    elif params.arch_type == "VGG11":
#        net = UNet11(channel_count=no_channels, pre_trained=True).cuda()
    else:
        
        print("Error: Architecture " + params.architecture + " not found.")
    
    
    # Xavier initialization of CNN weights
    
    if params.xav_init == 1:
        
        net.apply(weights_init)
    
    
    train_raw_path, train_mask_path, val_raw_path, val_mask_path, train_patients, val_patients = pathExtractorCrossValidation(k, m_paths, r_paths, patient_paths)
    
            
    # Training and validation data loaders

    loader_train, loader_val = get_data_loader(train_raw_path, train_mask_path, k, K, 
                                                data_path, val_raw_path, val_mask_path, 
                                                train_patients, val_patients)
    
    
    # Train and evaluate
    
    print("\nStart training network, fold " + str(k) + "...\n")

    loss, metrics_train, metrics_val, model_state, optimizer, optimizer_state = train.train(net, loader_train, loader_val, 
                                                                                            k, params.eval_frequency, 
                                                                                            params.I)
    
    
    # Save model and optimizer for later inference
    
    utilities.model_saving(model_state, optimizer_state, params.network_data_path, 'FinalTrainedWith' + params.train_with + '_' + params.prep_step + 'fold_' + str(k) + '.tar')
    
    losses.append(loss)

    metrics_names = np.unique(metrics_train_array[:,0])
    
    for i in range(len(metrics_train)):
        
        # Store arrays with mean and standard deviation results
        
        metrics_train_array[k,:,i] = np.array([float(metrics_train[i][1]),float(metrics_train[i][2])])
        
        metrics_val_array[k,:,i] = np.array([float(metrics_val[i][1]),float(metrics_val[i][2])])
        
    
    
     #Prints time estimate
     
    pred_time_seconds = (time.time()-starting_time)*K/(k+1)
     
    ETA = (datetime.datetime(curr_date.year,
        curr_date.month,
        curr_date.day,
        curr_time.hour, 
        curr_time.minute, 
        curr_time.second) + datetime.timedelta(seconds = pred_time_seconds)).strftime("%Y-%m-%d %H:%M:%S")
    
    print("Estimated time of Arrival: ", ETA)  



# Print cross-validation results and show figures
        
mean_loss = np.mean(np.array(losses))

std_loss = np.std(np.array(losses))

print('\nFinal {} loss over folds: {} +- {}\n'.format(params.loss_fun, mean_loss, std_loss))

fig = plt.figure(figsize = (13,5))

plt.bar(np.arange(1,K + 1),losses, color = 'b')

plt.title(params.loss_fun + ' loss over folds')

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

    plt.bar(np.arange(1,K + 1), metrics_train_array[:,0,i], color = 'b', yerr = metrics_train_array[:,1,i])
    
    plt.title(str(metrics_names[i]) + ' over training folds')
    
    plt.xlabel('Fold number')
    
    plt.ylabel(metrics_names[i])
    
    fig.savefig(params.network_data_path + metrics_names[i] + '_training_plot_' + str(K) + '_folds_' + 'trainedWith' + params.train_with + '_' + params.prep_step + '.png')
    
    
    # Results for validation
    
    fig = plt.figure(figsize = (13,5))

    plt.bar(np.arange(1,K + 1), metrics_val_array[:,0,i], color = 'r', yerr = metrics_val_array[:,1,i])
    
    plt.title(metrics_names[i] + ' over validation folds')
    
    plt.xlabel('Fold number')
    
    plt.ylabel(metrics_names[i])
    
    fig.savefig(params.network_data_path + metrics_names[i] + '_validation_plot_' + str(K) + '_folds_' + 'trainedWith' + params.train_with + '_' + params.prep_step + '.png')
    

    
#t1 = time.time()
#    
#dataset = QFlowDataset(r_paths[:,0].tolist(), m_paths[0], True, params.train_with, params.three_D, params.augmentation)
#
#test_loader = torch.utils.data.DataLoader(dataset, batch_size = 2, num_workers = 0, shuffle=True)
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
##
##x = np.squeeze(test_images[0].numpy())
##
##y = np.squeeze(test_images[1].numpy())
##
#x = test_images[0].numpy()
#
#y = test_images[1].numpy()
#
#print(x.shape, y.shape)
##
##center = y.shape[-1]//2
##
#plt.figure(figsize = (13,5))
#
#plt.subplot(131)
#
#plt.imshow(x[0,0,:,:], cmap = 'gray'), plt.colorbar()
#
#plt.subplot(132)
#
#plt.imshow(y[0,:,:], cmap = 'gray'), plt.colorbar()
#
#plt.subplot(133)
#
#plt.imshow(x[0,0,:,:]*y[0,:,:], cmap = 'gray'), plt.colorbar()

#
#plt.subplot(143)
#
#plt.imshow(y[0,:,:,20], cmap = 'gray'), plt.colorbar()
#
#plt.subplot(144)
#
#plt.imshow(x[0,:,:,20,0]*y[0,:,:,20], cmap = 'gray'), plt.colorbar()
