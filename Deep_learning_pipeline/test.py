#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:58:01 2020

@author: andres
"""

import sys

import os

import time

import datetime

import shutil

import torch

import numpy as np

import matplotlib.pyplot as plt

from datasets import QFlowDataset

from patientStratification import StratKFold

import params

import itertools

import utilities

import torch.optim as optim

import train

import models

import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import flowInformation

import png

from torch.autograd import Variable

from evaluate import Dice, Precision, Recall





class testing:
    
    """
    Provides testing of the pipeline.
    
    Params:
        
        - img_filename: filename(s) of image to test  [magnitude, phase]
        
        - img_path: folder where image, phase file and mask are located
        
        - mask_filename: corresponding mask filename (None if not available)
        
        - pha_filename: corresponding phase filename
        
        - model: model with which to test images
        
        - flow_path: folder with flow information (study dependent)
        
        - dest_path: folder where to save testing results (segmentations, MIPs...)
    
    
    Returns:
        
        - metric_results: results from evaluation (if mask is available)
    
        - flow_results: results from flow comparison
    
    """
    
    def __init__(self, img_filename, img_path, mask_filename, model_filename, model_path, flow_path, dest_path):
        
        self.img_filename = img_filename
        
        self.img_path = img_path
        
        self.mask_filename = mask_filename
        
        self.model_filename = model_filename
        
        self.model_path = model_path
        
        self.flow_path = flow_path
        
        self.dest_path = dest_path
    
    
    
    def readVTK(self, filename, order='F'):
            
        """
        Utility function to read vtk volume. 
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            - path: path where VTK file is located
            - filename: VTK file name
        
        Returns:
            
            - numpy array
            - data origin
            - data spacing
        
        """
    
        reader = vtk.vtkStructuredPointsReader()
    
        reader.SetFileName(self.img_path + filename)
    
        reader.Update()
    
        image = reader.GetOutput()
    
        numpy_array = vtk_to_numpy(image.GetPointData().GetScalars())
    
        numpy_array = numpy_array.reshape(image.GetDimensions(),order='F')
    
        numpy_array = numpy_array.swapaxes(0,1)
    
        origin = list(image.GetOrigin())
    
        spacing = list(image.GetSpacing())
    
        return numpy_array, origin, spacing
    
    
    
    
    def array2vtk(self, array, filename, origin = [0,0,0], spacing = [1,1,1]):
                
        """
        Convert array into .vtk file
        
        - Params:
            
            inherited class parameters (see description at beginning of the class)
            
            array: array to be converted into .vtk file
            
            filename: filename with which to save array as VTK file
            
            origin: origin of coordinate system, by default (0,0,0)
            
            spacing: spacing of coordinate system, by default (1,1,1)
        
        """
          
        vtk_writer = vtk.vtkStructuredPointsWriter()
    
            
        # Check if destination folder exists
        
        #print('Checking if destination folder exists\n')
            
        isdir = os.path.isdir(self.dest_path)
            
        if not isdir:
            
            os.makedirs(self.dest_path)
            
            print('Non-existing destination path. Created\n')
        
        # Check if files already exist in destination folder
            
        exist = filename in os.listdir(self.dest_path)
        
        overwrite = 'y'
        
        if exist:
            
            overwrite = input("File is already in folder. Do you want to overwrite? [y/n]\n")
        
        if overwrite == 'y' or overwrite == 'Y':
                
            vtk_writer.SetFileName(self.dest_path + filename)
                
            vtk_im = vtk.vtkStructuredPoints()
        
            vtk_im.SetDimensions((array.shape[1],array.shape[0],array.shape[2]))
            
            vtk_im.SetOrigin(origin)
            
            vtk_im.SetSpacing(spacing)
        
            pdata = vtk_im.GetPointData()
        
            vtk_array = numpy_to_vtk(array.swapaxes(0,1).ravel(order='F'),deep = 1, array_type=vtk.VTK_FLOAT)
        
            pdata.SetScalars(vtk_array)
        
            vtk_writer.SetFileType(vtk.VTK_BINARY)
        
            vtk_writer.SetInputData(vtk_im)
        
            vtk_writer.Update()
            
            #print('VTK file saved successfully!\n')
        
        else:
            
            print('\nOperation aborted\n')
    
    
    def Segmentation(self, X, out, Y, origin, spacing):
    
        """
        Provide a segmentation VTK file with testing results of mask overlapped
        with normal raw image.
        
        Params:
            
            - X: raw image tensor
            
            - out: output tensor from neural network
            
            - mask: provided mask tensor (if unavailable: None)
            
            - origin: origin of raw image to save as VTK
            
            - spacing: spacing of raw image to save as VTK
            
        
        Results:
            
            saved VTK image in color (G: correct segmentation B: under segmentation
            R: over segmentation)
        
        
        """
        
        final_result = np.zeros((X.shape[2], X.shape[3], X.shape[4], 3))
        
        
        if Y is not None:
            
            Y = Y.numpy()
            
        
        for i in range(final_result.shape[-1]):
            
            final_result[:,:,:,i] = 255.0*(X-np.amin(X))/(np.amax(X)-np.amin(X)) # Grayscale values from 0 to 255
        
    
        # If no mask available, leave segmented result in green
        
        if Y is None:
        
            ind_green = np.where(out > 0.5)
            
            out_aux = np.zeros(out.shape)
            
            out_aux[ind_green] = 255.0
            
            final_result[ind_green,1] = out_aux
            
            final_result[ind_green,0] = 0
            
            final_result[ind_green,2] = 0
        
        
        # If mask is available, leave coincident results in green, results with result but not with mask in red 
        # and results with mask but not with result in blue
        
        ind_pos_out = np.where(out > 0) # Result locations
        
        ind_pos_Y = np.where(Y > 0) # Mask locations
        
        out_aux = np.copy(out)
        
        Y_aux = np.copy(Y)
        
        out_aux[ind_pos_out] = 2 # Set result locations to 2
        
        Y_aux[ind_pos_Y] = 3 # Set mask locations to 3
        
        mult = out*Y
        
        ind_green = np.where(mult > 0)
        
        total = out_aux + Y_aux
        
        ind_red = np.where(total == 2)
        
        ind_blue = np.where(total == 3)
        
        final_result[ind_red,0]  = 255.0
        
        final_result[ind_red,1] = 0
        
        final_result[ind_red,2] = 0
        
        final_result[ind_green,0]  = 0
        
        final_result[ind_green,1] = 255.0
        
        final_result[ind_green,2] = 0
        
        final_result[ind_blue,0]  = 0
        
        final_result[ind_blue,1] = 0
        
        final_result[ind_blue,2] = 255.0
        
        if 'pha' in params.train_with:
        
            filename = self.img_filename[0].replace('pha','seg')
        
        elif not('pha' in params.train_with) and not('BF' in params.train_with):
            
            filename = self.img_filename[0].replace('mag','seg')
        
        elif not('pha' in params.train_with) and ('BF' in params.train_with):
            
            filename = self.img_filename[0].replace('magBF','seg')
        
        self.array2vtk(final_result, filename, origin, spacing)
    
    
    
    def MIP(self, X, out, Y):
        
        """
        Provide a MIP image with found result.
        
        Params:
            
            - X: raw image tensor
            
            - out: output tensor from neural network
            
            - mask: provided mask tensor (if unavailable: None)
            
        
        Results:
            
            saved PNG image in color (G: correct segmentation B: under segmentation
            R: over segmentation)
        
        
        """
        
        final_result = np.zeros((X.shape[2], X.shape[3], X.shape[4], 3))
        
        
        if Y is not None:
            
            Y = Y.numpy()
            
        
        for i in range(final_result.shape[-1]):
            
            final_result[:,:,:,i] = 255.0*(X-np.amin(X))/(np.amax(X)-np.amin(X)) # Grayscale values from 0 to 255
        
    
        # If no mask available, leave segmented result in green
        
        if Y is None:
        
            ind_green = np.where(out > 0.5)
            
            out_aux = np.zeros(out.shape)
            
            out_aux[ind_green] = 255.0
            
            final_result[ind_green,1] = out_aux
            
            final_result[ind_green,0] = 0
            
            final_result[ind_green,2] = 0
        
        
        # If mask is available, leave coincident results in green, results with result but not with mask in red 
        # and results with mask but not with result in blue
        
        ind_pos_out = np.where(out > 0) # Result locations
        
        ind_pos_Y = np.where(Y > 0) # Mask locations
        
        out_aux = np.copy(out)
        
        Y_aux = np.copy(Y)
        
        out_aux[ind_pos_out] = 2 # Set result locations to 2
        
        Y_aux[ind_pos_Y] = 3 # Set mask locations to 3
        
        mult = out*Y
        
        ind_green = np.where(mult > 0)
        
        total = out_aux + Y_aux
        
        ind_red = np.where(total == 2)
        
        ind_blue = np.where(total == 3)
        
        final_result[ind_red,0]  = 255.0
        
        final_result[ind_red,1] = 0
        
        final_result[ind_red,2] = 0
        
        final_result[ind_green,0]  = 0
        
        final_result[ind_green,1] = 255.0
        
        final_result[ind_green,2] = 0
        
        final_result[ind_blue,0]  = 0
        
        final_result[ind_blue,1] = 0
        
        final_result[ind_blue,2] = 255.0
        
        if 'pha' in params.train_with:
        
            filename = self.img_filename[0].replace('pha','seg')
            
            filename = filename.replace('vtk','png')
        
        elif not('pha' in params.train_with) and not('BF' in params.train_with):
            
            filename = self.img_filename[0].replace('mag','seg')
            
            filename = filename.replace('vtk','png')
        
        elif not('pha' in params.train_with) and ('BF' in params.train_with):
            
            filename = self.img_filename[0].replace('magBF','seg')
            
            filename = filename.replace('vtk','png')
            
        mip_result = np.zeros((X.shape[0], X.shape[1], 3))
        
        for i in range(final_result.shape[-1]):
            
            mip_result[:,:,i] = final_result.max(final_result[:,:,:,i], axis = -1)
            
        png.from_array(mip_result, mode = 'RGB').save(self.dest_path + filename)    
        
        
    
        
    
    def extractTensors(self):
        
        img_arrays = []
        
        for i in range(len(self.img_filename)):
            
            # Load images
            
            # Extract numpy array from image and mask filenames
        
            img_array, origin, spacing = self.readVTK(self.img_filename[i])
            
            img_arrays.append(img_array)
        
        
        # Adjust to adequate tensor dimensions
            
        img = np.zeros((1,img_array.shape[0], img_array.shape[1], img_array.shape[2], len(img_arrays)))
        
        for i in range(len(self.img_filename)):
            
            img[:,:,:,:,i] = img_arrays[i]
        
        X = Variable(torch.from_numpy(np.flip(img,axis = 0).copy())).float()
            
        X = X.permute(-1,0,1,2) # Channels first

        if self.mask_filename is not None:

            # Load mask array

            mask_array, _, _ =  self.readVTK(self.mask_filename)
            
            # Adjust to adequate tensor dimensions
            
            mask = np.expand_dims(mask_array, axis = 0)

            Y = Variable(torch.from_numpy(np.flip(mask,axis = 0).copy())).long()
            
            return X, Y, origin, spacing
        
        else:
            
            return X, origin, spacing
    
    
    def modelPreparation(self):
        
        """
        Prepares model and optimizer to load them for inference.
        
        Returns: loaded model and optimizer
        
        """
        
        if params.architecture == "UNet_with_Residuals":
        
            net = models.UNet_with_Residuals(2).cuda()
            
            print(net)
            
            utilities.print_num_params(net)
        
        else:
            
            print('Wrong architecture. Please introduce a valid architecture')
        
        
        
        
        
        if params.opt == 'Adam':
    
            optimizer = optim.Adam(net.parameters(), params.lr)
    
        elif params.opt == 'RMSprop':
            
            optimizer = optim.RMSprop(net.parameters(), params.lr)
        
        elif params.opt == 'SGD':
            
            optimizer = optim.SGD(net.parameters(), params.lr)
        
        else:
            
            print('\nWrong optimizer. Please define a valid optimizer (Adam/RMSprop/SGD)\n')
        
        
        net, optimizer = utilities.model_loading(net, optimizer, self.model_path, self.model_filename)
        
        return net, optimizer
        
    
    
    def flowFromMask(self, mask, raw, spacing):
        
        """
        Compute flow parameters from masks and phase images.
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            - mask: binary or quasi-binary 3D array with results from neural network segmentation
            - raw: corresponding phase image
            - spacing: pixel size, useful to extract area and flow information
        
        Returns:
            
            - result: 2D array with results on average velocity, standard deviation,
                    maximum velocity, minimum velocity, area, net flow, positive flow
                    and negative flow
        
        """
        
        # Multiply mask to raw phase image: get just ROI information
        
        mult = mask * raw
        
        result = np.zeros((8, mask.shape[2]))
        
        for j in range(mask.shape[2]): # Frame by frame analysis
            
            mult_frame = mult[:,:,j] # Frame of the multiplication operator
            
            s = np.sum(mult_frame.flatten()) # Overall sum of each frame. If > 0 --> left kidney, if < 0 --> right kidney
            
            ind = np.where(mult_frame != 0) # Indexes inside the ROI
            
            result[0,j] = np.abs(np.mean(mult_frame[ind].flatten()))/100 # Mean values
            
            result[1,j] = np.abs(np.std(mult_frame[ind].flatten()))/100 # Standard deviation
            
            result[2,j] = np.amax(mult_frame)/100 # Maximum value
            
            result[3,j] = np.amin(mult_frame)/100 # Minimum value
            
            result[4,j] = np.array(ind).shape[-1]*spacing[0]*spacing[1]/100 # Area
    
            result[5,j] = abs(s/result[4,j])/10000 # Net flow
            
            ind_pos = np.where(mult_frame > 0) # Positive voxel values
                
            ind_neg = np.where(mult_frame < 0) # Negative voxel values
            
            if s > 0:
        
                result[6,j] = np.sum(mult_frame[ind_pos].flatten())/(result[4,j]*10000) # Positive flow values
                
                if len(ind_neg) != 0: # Check if there are any negative voxels
                    
                    result[7,j] = np.sum(mult_frame[ind_neg].flatten())/(result[4,j]*10000) # Negative flow values
                
                else:
                    
                    result[7,j] = 0
            
            elif s < 0:
                
                result[6,j] = np.sum(mult_frame[ind_neg].flatten())/(result[4,j]*10000) # Positive flow values
                
                if len(ind_pos) != 0: # Check if there are any positive voxels
                    
                    result[7,j] = np.sum(mult_frame[ind_pos].flatten())/(result[4,j]*10000) # Negative flow values
                
                else:
                    
                    result[7,j] = 0
                
        return result
        


    def __main__(self):
        
        coincide = 1
        
        coincide_img = 1
        
        if not('CKD015' in self.img_filename[0]):
            
            # Avoid that patient (CKD015)
        
            # Make sure that image and mask coincide (if mask is not None)
            
            if self.mask_filename is not None:
                
                if (self.img_filename[0].replace('mag','msk') == self.mask_filename) or (self.img_filename[0].replace('pha','msk') == self.mask_filename) or (self.img_filename[0].replace('magBF','msk') == self.mask_filename):
                    
                    coincide = 1
                
                else:
                    
                    coincide = 0
                    
                    print('\nImage and mask files are not coincident. Please provide coincident image and mask files\n')
            
            
            # Make sure that if there is more than one image filename, these names are correspondent
            
            if len(self.img_filename) > 1:
                
                coincide_img = 0
                
                if self.img_filename[1].replace('pha', 'mag') == self.img_filename[0] or self.img_filename[1].replace('pha', 'magBF') == self.img_filename[0]:
                    
                    coincide_img = 1
                
                
                else:
                    
                    coincide_img = 0
                    
                    print('\n Magnitude and phase files are not coincident. Please provide coincident image and mask files\n')
                    
            
                
            
            if coincide == 1 and coincide_img == 1:
                
                # All files are coincident. Extract image and mask tensors (mask tensor if mask available)
                
                if self.mask_filename is not None:
                    
                    X,Y, origin, spacing = self.extractTensors()
                
                else:
                    
                    X, origin, spacing = self.extractTensors()
                    
                    Y = None
                
                
                model, optimizer = self.modelPreparation()
                
                if params.three_D: # 2D + time architecture
                    
                    out = self.model(X.cuda(non_blocking=True)).data
                    
                    out = torch.argmax(out, 1).cpu() # Inference output
                    
                else: # 2D architecture
                    
                    out = torch.zeros(1, X.shape[-3], X.shape[-2], X.shape[-1])
                    
                    for i in range(X.shape[-1]):
                        
                        out_aux = self.model(X[:,:,:,:,i].cuda(non_blocking=True)).data
                        
                        out[:,:,:,i] = torch.argmax(out_aux, 1).cpu() # Inference output
                
                if self.mask_filename is not None:
                    
                    metric_results = []
                    
                    # Extract segmentation metrics
                    
                    for metric in params.metrics:
                    
                        if metric == 'Dice' or metric == 'dice' or metric == 'DICE':
                            
                            dice = Dice(Y, out)
                            
                            metric_results.append(dice.online())
                            
                            print('Dice coefficient for {}: {}\n'.format(self.img_filename[0].split('/')[-1], dice.online()))
                        
                        
                        elif metric == 'Precision' or metric == 'PRECISION' or metric == 'precision':
                            
                            prec = Precision(Y, out)
                            
                            metric_results.append(prec.online())
                            
                            print('Precision for {}: {}\n'.format(self.img_filename[0].split('/')[-1], prec.online()))
                        
                        
                        elif metric == 'Recall' or metric == 'recall' or metric == 'RECALL':
                            
                            rec = Recall(Y, out)
                            
                            metric_results.append(rec.online())
                            
                            print('Recall for {}: {}\n'.format(self.img_filename[0].split('/')[-1], rec.online()))
                    
                # Provide resulting segmentations and MIPs
                
                self.Segmentation(X[0,0,:,:,:].numpy(), out.numpy(), Y, origin, spacing)
                
                self.MIP(X[0,0,:,:,:].numpy(), out.numpy(), Y)
                
                # Extract flow information from outputted result
                
                if 'pha' in params.train_with:
                    
                    pha_array = X[0,0,:,:,:]
                
                elif 'both' in params.train_with:
                    
                    pha_array = X[0,1,:,:,:]
                
                else:
                    
                    if 'BF' in params.train_with:
                    
                        pha_filename = self.img_filename[0].replace('magBF','pha')
                    
                    else:

                        pha_filename = self.img_filename[0].replace('mag','pha')
                        
                    pha_array, origin, spacing = self.readVTK(pha_filename)
                
                flow_out = self.flowFromMask(out, pha_array, spacing)
                
                return flow_out
                
#                if self.mask_filename is not None: # Extract flow information from MAT file
#                    
#                    # Look for corresponding TXT filename
#                    
#                    txt_files = sorted(os.listdir(self.flow_path))
#                    
#                    txt_filename = self.mask_filename.replace('msk',)





test = testing('CKD2_CKD019_MRI3_dx_mag_0.vtk', '/home/andres/Documents/_Data/_Patients/_Raw/_ckd2/CKD019_MRI3/', 
               'CKD2_CKD019_MRI3_dx_msk_0.vtk', 'FinalTrainedWithmag__rawfold_3.tar', '/home/andres/Documents/_Data/Network_data/UNet_with_Residuals/',
               '/home/andres/Documents/_Data/CKD_Part2/4_Flow/','/home/andres/Documents/_Results/Test_09March')

flow = test.__main__()