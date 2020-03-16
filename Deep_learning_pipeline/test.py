#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:58:01 2020

@author: andres
"""

import sys

import os

import time

import torch

import numpy as np

import matplotlib.pyplot as plt

import params

import utilities

import torch.optim as optim

import models

import pandas as pd

import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from flowInformation import FlowInfo

from torch.autograd import Variable

from evaluate import Dice, Precision, Recall

import cv2



class testing:
    
    """
    Provides testing of the pipeline.
    
    Params:
        
        - img_filename: list of filename(s) of image to test [magnitude]/[phase]/[magnitude, phase]
        
        - img_path: list of folder(s) where image, phase file and mask are located
        
        - mask_filename: corresponding list of mask filenames (None if not available)
        
        - pha_filename: corresponding phase filename
        
        - model: model with which to test images
        
        - flow_path: folder with flow information (study dependent)
        
        - dest_path: folder where to save testing results (segmentations, MIPs...)
        
        - excel_file: file with Excel measurements (if available)
    
    
    Returns:
        
        - metric_results: results from evaluation (if mask is available)
    
        - flow_results: results from flow comparison
    
    """
    
    def __init__(self, study, img_filename, img_path, mask_filename, 
                 model_filename, model_path, flow_path, dest_path, excel_file):
        
        self.study = study
        
        self.img_filename = img_filename
        
        self.img_path = img_path
        
        self.mask_filename = mask_filename
        
        self.model_filename = model_filename
        
        self.model_path = model_path
        
        self.flow_path = flow_path
        
        self.dest_path = dest_path
        
        self.excel_file = excel_file
    
    
    
    def readVTK(self, img_path, filename, order='F'):
            
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
    
        reader.SetFileName(img_path + filename)
    
        reader.Update()
    
        image = reader.GetOutput()
    
        numpy_array = vtk_to_numpy(image.GetPointData().GetScalars())
    
        numpy_array = numpy_array.reshape(image.GetDimensions(),order='F')
    
        numpy_array = numpy_array.swapaxes(0,1)
    
        origin = list(image.GetOrigin())
    
        spacing = list(image.GetSpacing())
    
        return numpy_array, origin, spacing
    
    
    
    
#    def array2vtk(self, array, filename, origin = [0,0,0], spacing = [1,1,1]):
#                
#        """
#        Convert array into .vtk file
#        
#        - Params:
#            
#            inherited class parameters (see description at beginning of the class)
#            
#            array: array to be converted into .vtk file
#            
#            filename: filename with which to save array as VTK file
#            
#            origin: origin of coordinate system, by default (0,0,0)
#            
#            spacing: spacing of coordinate system, by default (1,1,1)
#        
#        """
#          
#        vtk_writer = vtk.vtkStructuredPointsWriter()
#    
#            
#        # Check if destination folder exists
#        
#        #print('Checking if destination folder exists\n')
#            
#        isdir = os.path.isdir(self.dest_path)
#            
#        if not isdir:
#            
#            os.makedirs(self.dest_path)
#            
#            print('Non-existing destination path. Created\n')
#        
#        # Check if files already exist in destination folder
#            
#        exist = filename in os.listdir(self.dest_path)
#        
#        overwrite = 'y'
#        
#        if exist:
#            
#            overwrite = input("File is already in folder. Do you want to overwrite? [y/n]\n")
#        
#        if overwrite == 'y' or overwrite == 'Y':
#                
#            vtk_writer.SetFileName(self.dest_path + filename)
#                
#            vtk_im = vtk.vtkStructuredPoints()
#        
#            vtk_im.SetDimensions((array.shape[1],array.shape[0],array.shape[2], array.shape[3]))
#            
#            vtk_im.SetOrigin(origin)
#            
#            vtk_im.SetSpacing(spacing)
#        
#            pdata = vtk_im.GetPointData()
#        
#            vtk_array = numpy_to_vtk(array.swapaxes(0,1).ravel(order='F'),deep = 1, array_type=vtk.VTK_FLOAT)
#        
#            pdata.SetScalars(vtk_array)
#        
#            vtk_writer.SetFileType(vtk.VTK_BINARY)
#        
#            vtk_writer.SetInputData(vtk_im)
#        
#            vtk_writer.Update()
#            
#            #print('VTK file saved successfully!\n')
#        
#        else:
#            
#            print('\nOperation aborted\n')
    
    
    def Segmentation(self, X, out, Y, img_filename):
    
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
        
        final_result = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3))
        
        
        if Y is not None:
            
            Y = Y.numpy()
            
        
        for i in range(final_result.shape[-1]):
            
            final_result[:,:,:,i] = (X-np.amin(X))/(np.amax(X)-np.amin(X)) # Grayscale values from 0 to 255
        
            final_result[:,:,:,i].astype(int)
    
        
        # If no mask available, leave segmented result in green
        
        if Y is None:
        
            ind_green = np.where(out > 0.5)
            
            out_aux = np.zeros(out.shape)
            
            out_aux[ind_green] = 1
            
            final_result[ind_green[1,:], ind_green[2,:], ind_green[3,:], 0] = 0
        
            final_result[ind_green[1,:], ind_green[2,:], ind_green[3,:], 1] = 1
            
            final_result[ind_green[1,:], ind_green[2,:], ind_green[3,:], 2] = 0
        
        
        # If mask is available, leave coincident results in green, results with result but not with mask in red 
        # and results with mask but not with result in blue
        
        ind_pos_out = np.where(out > 0) # Result locations
        
        ind_pos_Y = np.where(Y > 0) # Mask locations
        
        out_aux = np.copy(out)
        
        Y_aux = np.copy(Y)
        
        out_aux[ind_pos_out] = 2 # Set result locations to 2
        
        Y_aux[ind_pos_Y] = 3 # Set mask locations to 3
        
        mult = out*Y
        
        ind_green = np.array(np.where(mult > 0))
        
        total = out_aux + Y_aux
        
        ind_red = np.array(np.where(total == 2))
        
        ind_blue = np.array(np.where(total == 3))

        # Color final mask result
        
        final_result[ind_red[1,:], ind_red[2,:], ind_red[3,:], 0]  = 1
        
        final_result[ind_red[1,:], ind_red[2,:], ind_red[3,:], 1]  = 0
        
        final_result[ind_red[1,:], ind_red[2,:], ind_red[3,:], 2]  = 0
        
        final_result[ind_green[1,:], ind_green[2,:], ind_green[3,:], 0] = 0
        
        final_result[ind_green[1,:], ind_green[2,:], ind_green[3,:], 1] = 1
        
        final_result[ind_green[1,:], ind_green[2,:], ind_green[3,:], 2] = 0
        
        final_result[ind_blue[1,:], ind_blue[2,:], ind_blue[3,:], 0] = 0
        
        final_result[ind_blue[1,:], ind_blue[2,:], ind_blue[3,:], 1] = 0
        
        final_result[ind_blue[1,:], ind_blue[2,:], ind_blue[3,:], 2] = 1
        
        return final_result
        
#        
#        if 'pha' in params.train_with:
#        
#            filename = img_filename[0].replace('pha','seg')
#        
#        elif not('pha' in params.train_with) and not('BF' in params.train_with):
#            
#            filename = img_filename[0].replace('mag','seg')
#        
#        elif not('pha' in params.train_with) and ('BF' in params.train_with):
#            
#            filename = img_filename[0].replace('magBF','seg')
        
        
        #final_result = final_result/np.amax(final_result)
        
#        plt.figure(figsize = (13,5))
#        
#        plt.imshow(final_result[:,:,35,:])
        
#        plt.subplot(131)
#        
#        plt.imshow(final_result[:,:,0,0], cmap = 'gray')
#        
#        plt.subplot(132)
#        
#        plt.imshow(final_result[:,:,0,1], cmap = 'gray')
#        
#        plt.subplot(133)
#        
#        plt.imshow(final_result[:,:,0,2], cmap = 'gray')
        
#        filename = filename.replace('vtk','nrrd')
#        
#        
#        # Save result as .nrrd. Permute dimensions if I finally do this!!
#        
#        segm = sitk.GetImageFromArray(self.dest_path + final_result)
#        
#        writer = sitk.ImageFileWriter()
#        
#        writer.SetFileName(filename)
#        
#        writer.Execute(segm)
#        
        #self.array2vtk(final_result, filename, origin, spacing)
    
    
    
    def MIP(self, X, out, Y, img_filename):
        
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
        
        final_result = self.Segmentation(X, out, Y, img_filename[0])
        
        mip_result = np.round(np.sum(255*final_result, axis = -2)/final_result.shape[2]).astype(int)

        # Decide on final filename
        
        if 'pha' in params.train_with:
        
            filename = img_filename[0].replace('pha','mip')
        
        elif not('pha' in params.train_with) and not('BF' in params.train_with):
            
            filename = img_filename[0].replace('mag','mip')
        
        elif not('pha' in params.train_with) and ('BF' in params.train_with):
            
            filename = img_filename[0].replace('magBF','mip')
        
        filename = filename.replace('vtk','png')
        
        # Exchange blue and red channels to write PNG file with CV2
        
        mip_aux = np.copy(mip_result)
        
        mip_aux[:,:,0] = mip_result[:,:,-1]
        
        mip_aux[:,:,-1] = mip_result[:,:,0]

        cv2.imwrite(self.dest_path + filename, mip_aux)

        
        
    
        
    
    def extractTensors(self, img_filename, img_path, mask_filename):
        
        """
        Provide torch tensors with data. If the evaluation is in 2D, a list
        with the sums of the frames along time is outputted, too
        
        """
        
        img_arrays = []
        
        sum_t_list = []
        
        for i in range(len(img_filename)):
            
            # Load images
            
            # Extract numpy array from image and mask filenames
        
            img_array, origin, spacing = self.readVTK(img_path, img_filename[i])
            
            img_arrays.append(img_array)
            
            if not(params.three_D):
        
                sum_t_list.append(np.sum(img_array, axis = -1))
            
            
        
        # Adjust to adequate tensor dimensions
        
        # Compute number of channels
        
            
        if 'both' in params.train_with:
            
            channels = 2
        
        else:
            
            channels = 1
                
            
        img = np.zeros((1,img_array.shape[0], img_array.shape[1], img_array.shape[2], channels))
        
        for i in range(len(img_filename)):
            
            img[:,:,:,:,i] = img_arrays[i]
            
            
        
        X = Variable(torch.from_numpy(np.flip(img,axis = 0).copy())).float()
            
        X = X.permute(0, -1, 1, 2,-2) # Channels first

        if self.mask_filename is not None:

            # Load mask array

            mask_array, _, _ =  self.readVTK(img_path, mask_filename)
            
            # Adjust to adequate tensor dimensions
            
            mask = np.expand_dims(mask_array, axis = 0)

            Y = Variable(torch.from_numpy(np.flip(mask,axis = 0).copy())).long()
            
            return X, Y, origin, spacing, sum_t_list
        
        else:
            
            return X, origin, spacing, sum_t_list
        
        
    
    
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
            - raw: corresponding velocity array
            - spacing: pixel size, useful to extract area and flow information
        
        Returns:
            
            - result: 2D array with results on average velocity, standard deviation,
                    maximum velocity, minimum velocity, area, net flow, positive flow
                    and negative flow
        
        """
        
        # Multiply mask to raw phase image: get just ROI information
        
        mult = mask[0,:,:,:] * raw
        
        result = np.zeros((8, mask.shape[-1]))
        
        for j in range(mask.shape[-1]): # Frame by frame analysis
            
            mult_frame = mult[:,:,j] # Frame of the multiplication operator
            
            s = np.sum(mult_frame.flatten()) # Overall sum of velocities for each frame. If > 0 --> left kidney, if < 0 --> right kidney
            
            ind = np.where(mult_frame != 0) # Indexes inside the ROI
            
            result[0,j] = np.mean(mult_frame[ind].flatten()) # Mean values (cm/s)
            
            result[1,j] = np.std(mult_frame[ind].flatten()) # Standard deviation (cm/s)
            
            result[2,j] = np.amax(mult_frame) # Maximum value (cm/s)
            
            result[3,j] = np.amin(mult_frame) # Minimum value (cm/s)
            
            result[4,j] = (np.array(ind).shape[-1])*spacing[0]*spacing[1]/100 # Area (cm2)
    
            result[5,j] = result[0,j]/result[4,j] # Net flow (cm3/s = ml/s)
            
            ind_pos = np.where(mult_frame > 0) # Positive voxel values
                
            ind_neg = np.where(mult_frame < 0) # Negative voxel values
            
            if s > 0:
        
                result[6,j] = np.mean(mult_frame[ind_pos].flatten())/(result[4,j]) # Positive flow values
                
                if len(ind_neg) != 0: # Check if there are any negative voxels
                    
                    result[7,j] = np.mean(mult_frame[ind_neg].flatten())/(result[4,j]) # Negative flow values
                
                else:
                    
                    result[7,j] = 0
            
            elif s < 0:
                
                result[6,j] = np.mean(mult_frame[ind_neg].flatten())/(result[4,j]) # Positive flow values
                
                if len(ind_pos) != 0: # Check if there are any positive voxels
                    
                    result[7,j] = np.mean(mult_frame[ind_pos].flatten())/(result[4,j]) # Negative flow values
                
                else:
                    
                    result[7,j] = 0
                
        return result
    
    
    def phase2velocity(self, phase_array, img_filename):
        
        """
        Transforms raw phase image into velocity map, to later extract flow information.
        
        Params:
            
            - phase array: array with phase values
        
        Returns:
        
            -vel_array: array with velocity values
            
            
        """
        
        venc = 100 # Decide later on how to read this!!
        
        if 'CKD' in self.study:
        
            # Analyze CKD1 and CKD2 studies
            
            vel_array = phase_array/venc
            
            
        
        elif self.study == 'hero':
            
            # Analyze Heroic study
            
            if 'venc080' in img_filename:
                
                venc = 80
            
            elif 'venc120' in img_filename:
                
                venc = 120
            
            else:
                
                venc = 100
            
            min_pha = np.amin(phase_array)
            
            max_pha = np.amax(phase_array)
            
            if max_pha - min_pha < 2: # Range probably between 0 and 1
                
                aux_array = 2*phase_array - 1
                
                vel_array = aux_array*venc
            
            else:
                
                aux_array = phase_array/(np.amax(phase_array))
                
                vel_array = aux_array*venc
        
        elif self.study == 'extr':
            
            # Analyze extra images
            
            if '20190103' in img_filename:
                
                aux_array = phase_array/(np.amax(phase_array))
                
                vel_array = aux_array*venc
            
            else:
                
                vel_array = phase_array/venc
            
        return vel_array
        
    
    def excelInfo(self, img_filename):
        
        """
        Extract flow information from Excel file.
        
        
        Returns:
            
            - array with maximum velocity, minimum velocity and mean arterial flow
        
        
        """
    
        # Extract measured patients from CKD1
    
        target_patient = img_filename[5:11]
        
        target_rep = img_filename[12:16]
        
        target_orient = img_filename[17:19]
        
        df = pd.read_excel(self.flow_path + self.excel_file) # can also index sheet by name or fetch all sheets
        
        flow = np.array(df['Mean_arterial_flow'].tolist()) # List with flow values
        
        max_v = np.array(df['Peak_velocity_max'].tolist()) # List with maximum velocity values
        
        min_v = np.array(df['Peak_velocity_min'].tolist()) # List with minimum velocity flow values
        
        rep = df['MR_Visit'].tolist() # List with repetitions
        
        patients = df['Subject_ID'].tolist()
        
        orient_ckd1 = np.array(df['left_right_kidney'].tolist()) 
        
        ind_left = np.where(orient_ckd1 == 'left')
        
        ind_right = np.where(orient_ckd1 == 'right')
        
        orient_final = np.copy(orient_ckd1) # List with kidney orientations
        
        orient_final[ind_left] = 'si'
        
        orient_final[ind_right] = 'dx' 
        
        # Patient measurements

        patient_ind = [i for i, s in enumerate(patients) if target_patient in s] # Indexes of same patient
        
        orients = orient_final[patient_ind] # Patient orientations
        
        ind_orient = [i for i, s in enumerate(orients) if target_orient in s]
        
        patient_orient_ind = patient_ind[ind_orient] # Indexes of patient with same kidney orientation
        
        patient_rep = rep[patient_orient_ind] # Patient visits with same orientation

        ind_rep = [i for i, s in enumerate(patient_rep) if target_rep in s]
        
        patient_orient_rep_ind = patient_ind[ind_rep]
        
        return flow[patient_orient_rep_ind], max_v[patient_orient_rep_ind], min_v[patient_orient_rep_ind]
        
    

    def __main__(self):
        
        coincide = 1
        
        coincide_img = 1
        
        cont = 0
        
        all_results = []
        
        result_flows = []
        
        gt_flows = []
        
        for mask_path in self.mask_filename:
        
            if not('CKD015' in self.img_filename[0]):
                
                # Avoid that patient (CKD015)
            
                # Make sure that image and mask coincide (if mask is not None)
                
                if len(self.mask_filename) != 0:
                    
                    if (self.img_filename[cont][0].replace('mag','msk') == self.mask_filename[cont]) or (self.img_filename[cont][0].replace('pha','msk') == self.mask_filename[cont]) or (self.img_filename[cont][0].replace('magBF','msk') == self.mask_filename[cont]):
                        
                        coincide = 1
                    
                    else:
                        
                        coincide = 0
                        
                        print('\nImage and mask files are not coincident. Please provide coincident image and mask files\n')
                
                
                # Make sure that if there is more than one image filename, these names are correspondent
                
                if len(self.img_filename[cont]) > 1:
                    
                    coincide_img = 0
                    
                    if self.img_filename[cont][1].replace('pha', 'mag') == self.img_filename[cont][0] or self.img_filename[cont][1].replace('pha', 'magBF') == self.img_filename[cont][0]:
                        
                        coincide_img = 1
                    
                    
                    else:
                        
                        coincide_img = 0
                        
                        print('\n Magnitude and phase files are not coincident. Please provide coincident image and mask files\n')
                        
                
                    
                
                if coincide == 1 and coincide_img == 1:
                    
                    # All files are coincident. Extract image and mask tensors (mask tensor if mask available)
                    
                    if len(self.mask_filename) != 0:
                        
                        if params.three_D:
                        
                            X,Y, origin, spacing, _ = self.extractTensors(self.img_filename[cont], self.img_path[cont], self.mask_filename[cont])
                        
                        else:
                            
                            X,Y, origin, spacing, sum_t_list = self.extractTensors(self.img_filename[cont], self.img_path[cont], self.mask_filename[cont])
                    
                    else:
                        
                        if params.three_D:
                        
                            X, origin, spacing, _ = self.extractTensors(self.img_filename[cont], self.img_path[cont], None)
                        
                        else:
                            
                            X, origin, spacing, sum_t_list = self.extractTensors(self.img_filename[cont], self.img_path[cont], None)
                        
                        Y = None
                    
                    
                    with torch.no_grad():
                    
                        model, optimizer = self.modelPreparation()
                        
                        model.eval() # Model in evaluation mode
                        
                        if params.three_D: # 2D + time architecture
                            
                            out = model(X.cuda(non_blocking=True)).data
                            
                            out = torch.argmax(out, 1).cpu() # Inference output
                            
                        else: # 2D architecture
                            
                            out = torch.zeros(1, X.shape[-3], X.shape[-2], X.shape[-1])
                            
                            for i in range(X.shape[-1]):
                                
                                X_aux = torch.zeros(X.shape[0], X.shape[1] + len(sum_t_list), X.shape[2], X.shape[3])
                                
                                X_aux[:, :X.shape[1],:,:] = X[:,:,:,:,i]
                                
                                X_aux[:, X.shape[1]:,:,:] = torch.tensor(np.array(sum_t_list))
                                
                                out_aux = model(X_aux.cuda(non_blocking=True)).data
    
                                out[:,:,:,i] = torch.argmax(out_aux, 1).cpu() # Inference output
                        
#                        plt.figure(figsize = (13,5))
#                        
#                        plt.subplot(131)
#                        
#                        plt.imshow(X_aux[0,0,130:158,130:158], cmap = 'gray')
#                        
#                        plt.subplot(132)
#                        
#                        plt.imshow(out[0,130:158,130:158,-1], cmap = 'gray')
#                        
#                        plt.subplot(133)
#                        
#                        plt.imshow(Y[0,130:158,130:158,-1], cmap = 'gray')
                        
                        if self.mask_filename is not None:
                            
                            metric_results = []
 
                            # Extract segmentation metrics
                            
                            for metric in params.metrics:
                            
                                if metric == 'Dice' or metric == 'dice' or metric == 'DICE':
                                    
                                    dice = Dice(Y, out)
                                    
                                    metric_results.append(dice.online())
                                    
                                    print('Dice coefficient for {}: {}\n'.format(self.img_filename[cont][0].split('/')[-1], dice.online()))
                                
                                
                                elif metric == 'Precision' or metric == 'PRECISION' or metric == 'precision':
                                    
                                    prec = Precision(Y, out)
                                    
                                    metric_results.append(prec.online())
                                    
                                    print('Precision for {}: {}\n'.format(self.img_filename[cont][0].split('/')[-1], prec.online()))
                                
                                
                                elif metric == 'Recall' or metric == 'recall' or metric == 'RECALL':
                                    
                                    rec = Recall(Y, out)
                                    
                                    metric_results.append(rec.online())
                                    
                                    print('Recall for {}: {}\n'.format(self.img_filename[cont][0].split('/')[-1], rec.online()))
                                
                            all_results.append(metric_results)

                            
                        # Provide resulting segmentations and MIPs
                        
                        #self.Segmentation(X[0,0,:,:,:].numpy(), out.numpy(), Y, origin, spacing, self.img_filename[cont])
                        
                        self.MIP(X[0,0,:,:,:].numpy(), out.numpy(), Y, self.img_filename[cont])
                        
                        # Extract flow information from outputted result
                        
                        if 'pha' in params.train_with:
                            
                            pha_array = X[0,0,:,:,:]
                        
                        elif 'both' in params.train_with:
                            
                            pha_array = X[0,1,:,:,:]
                        
                        else:
                            
                            if 'BF' in params.train_with:
                            
                                pha_filename = self.img_filename[cont][0].replace('magBF','pha')
                            
                            else:
        
                                pha_filename = self.img_filename[cont][0].replace('mag','pha')
                                
                            pha_array, origin, spacing = self.readVTK(self.img_path[cont], pha_filename)
                        
                            # Obtain array with velocities (cm/s)
                            
                            vel_array = self.phase2velocity(pha_array, self.img_filename[cont][0])
                            
                            flow_out = self.flowFromMask(out.numpy(), vel_array, spacing)
                            
                            result_flows.append(flow_out)
                        
                        # Extract flow information from ground truth
                        
                        if self.study == 'CKD1':
                            
                            
                            ref_flow, ref_max_v, ref_min_v = self.excelInfo(self.img_filename[cont][0])
                            
                            result_flow = np.mean(result_flows[-1][-3])
                            
                            result_v_max = np.max(result_flows[-1][0])
                            
                            result_v_min = np.min(result_flows[-1][0])
                            
                            gt_flows.append([ref_flow, ref_max_v, ref_min_v])
                            
                            # Save results in bar plots
                            
                            
                            fig = plt.figure(figsize = (13,5))
                            
                            plt.bar(np.arange(1), [ref_flow, result_flow], 'b')
                            
                            plt.title('Net flow comparison for' + str(self.img_filename[cont][0]))
                            
                            plt.xlabel('Reference and result flows')
                            
                            plt.ylabel('Flow (ml/s)')
                            
                            fig.savefig(self.dest_path + 'flow_comparison' + str(self.img_filename[cont][0]) + '.png')
                            
                            
                            fig = plt.figure(figsize = (13,5))
                            
                            plt.bar(np.arange(1), [ref_max_v, result_v_max], 'b')
                            
                            plt.title('v_max comparison for' + str(self.img_filename[cont][0]))
                            
                            plt.xlabel('Reference and result v_max')
                            
                            plt.ylabel('v_max (cm/s)')
                            
                            fig.savefig(self.dest_path + 'v_max_comparison' + str(self.img_filename[cont][0]) + '.png')
                            
                            
                            fig = plt.figure(figsize = (13,5))
                            
                            plt.bar(np.arange(1), [ref_min_v, result_v_min], 'b')
                            
                            plt.title('v_min comparison for' + str(self.img_filename[cont][0]))
                            
                            plt.xlabel('Reference and result v_min')
                            
                            plt.ylabel('v_min (cm/s)')
                            
                            fig.savefig(self.dest_path + 'v_min_comparison' + str(self.img_filename[cont][0]) + '.png')
                            
                        
                        
                        else:
                            
                            # Extract flow information from txt_file
                            
                            # Search corresponding txt file
                            
                            txt_files = os.listdir(self.flow_path)
                            
                            # Get index of corresponding txt file
                            
                            # Get orientation and repetition number of file
                            
                            if 'dx' in self.img_filename[cont][0]:
                                
                                orient = 'dx'
                            
                            
                            elif 'si' in self.img_filename[cont][0]:
                                
                                orient = 'si'
                                
                                
                                
                            if '_0' in self.img_filename[cont][0]:
                                
                                rep = '_0'
                            
                            
                            elif '_1' in self.img_filename[cont][0]:
                                
                                rep = '_1'
                            
                            
                            ind = self.img_filename[cont][0].index(orient)
                                
                            
                            ind_txt = [i for i, s in enumerate(txt_files) if self.img_filename[cont][0][:ind + 2] in s]
                            
                            ind_final = -1
                            
                            
                            for k in ind_txt:
                                
                                if not('venc' in self.img_filename[cont][0]):
                                
                                    if rep in txt_files[k]:
                                        
                                        ind_final = k
                                
                                else:
                                    
                                    if ('venc080' in txt_files[k] and 'venc080' in self.img_filename[cont][0]) and (rep in txt_files[k]):
                                        
                                        ind_final = k
                                    
                                    elif 'venc100' in txt_files[k] and 'venc100' in self.img_filename[cont][0] and (rep in txt_files[k]):
                                        
                                        ind_final = k
                                        
                                    elif 'venc120' in txt_files[k] and 'venc120' in self.img_filename[cont][0] and (rep in txt_files[k]):
                                        
                                        ind_final = k
                                    
                                    
                            if k == -1:
                                
                                print('Flow file not found')
                                
                                exit()

    
                            load_info_mat = FlowInfo(self.study, self.flow_path, None, None, 'load', True, txt_files[ind_final])
                            
                            mean_v, std_v, max_v, min_v, energy, area, net_flow, pos_flow, neg_flow = load_info_mat.__main__()
                            
                            fig = plt.figure(figsize = (13,5))
                            
                            plt.plot(np.arange(1,vel_array.shape[-1] + 1), mean_v/np.max(np.abs(mean_v)), 'b', label = 'Ground truth')
                            
                            plt.plot(np.arange(1,vel_array.shape[-1] + 1), result_flows[-1][0]/np.max(np.abs(result_flows[-1][0])), 'r', label = 'Result')
                            
                            plt.title('Normalized flow comparison')
                            
                            plt.xlabel('Time frame #')
                                       
                            plt.ylabel('Normalized flow')
                            
                            plt.legend()
                            
                            fig.savefig(self.dest_path + 'flow_comparison' + str(self.img_filename[cont][0][:-3]) + '.png')
       
        
                            # Save segmentation results
                            
                            results_array = np.array(metric_results)
                            
                            txt_filename = self.dest_path + self.mask_filename[cont].replace('msk','metrics')
                            
                            txt_filename = txt_filename.replace('vtk','txt')
                            
#                            with open(txt_filename, 'a') as file:
#                                
#                                for metric in params.metrics:
#                    
#                                    file.write(metric + ' ')
#                                
#                                file.write('\n')
#                                
#                                for i in range(len(results_array)):
#                                    
#                                    if len(params.metrics) == 1:
#                                        
#                                        file.write(str(results_array[i]) + ' ')
#                                    
#                                    elif len(params.metrics) > 1:
#                                        
#                                        file.write(str(results_array[i][j]) + ' ')
#                                    
#                                    file.write('\n')
                            
                            
                            cont += 1

            return flow_out
                
#                if self.mask_filename is not None: # Extract flow information from MAT file
#                    
#                    # Look for corresponding TXT filename
#                    
#                    txt_files = sorted(os.listdir(self.flow_path))
#                    
#                    txt_filename = self.mask_filename.replace('msk',)





test = testing('CKD2',[['CKD2_CKD112_MRI3_dx_magBF_0.vtk']], ['/home/andres/Documents/_Data/_Patients/_Raw/_ckd2/CKD112_MRI3/'], 
               ['CKD2_CKD112_MRI3_dx_msk_0.vtk'], 'trainedWithmagBF_rawfold_0.tar', '/home/andres/Documents/_Data/Network_data/UNet_with_Residuals/',
               '/home/andres/Documents/_Data/CKD_Part2/4_Flow/','/home/andres/Documents/_Results/Test_09March/', None)

flow = test.__main__()

