#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:58:01 2020

@author: andres
"""

### import sys

import os

import time

import torch

import numpy as np

import matplotlib.pyplot as plt

import params

import utilities

import itertools

import torch.optim as optim

import models

import pandas as pd

import vtk

import itertools

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from flowInformation import FlowInfo

from torch.autograd import Variable

from evaluate import Dice, Precision, Recall

import cv2

import evaluate

from datasets import QFlowDataset

import models

import re

import flowStatistics


class testing:
    
    
    """
    Perform testing on cross-validation, testing or CKD1 test datasets, given the folder where the test dataset is located, the set of models with which to perform the inference and extra data for the extraction of flows
    
    Params:
    
        - test_path: path where the test set is allocated (str)
        
        - model_filenames: list with the filenames of the models to test (list of str)
        
        - model_path: folder where the model files have been saved (str)
        
        - flow_paths: list of folders with biomarker information (list of str)
        
        - excel_file: excel file with flow information from CKD1 study (str)
        
        - dest_path: folder where to save results as MIPs or biomarker comparisons (str)
        
        
    Return:
    
        - result_flows: biomarker values from the network and from outside (go later into a statistical framework) (list of arrays)
        
        - result_metrics: if masks are available, metric results are also saved (Dice, precision, recall) (list of arrays)
    
        - files: 2D+time file names that have been evaluated (list of str)
        
    """

    
    def __init__(self, test_path, model_filenames, model_path, flow_paths, excel_file, dest_path):
        
        
        self.test_path = test_path
        
        self.model_filenames = model_filenames
        
        self.model_path = model_path
        
        self.flow_paths = flow_paths
        
        self.excel_file = excel_file
        
        self.dest_path = dest_path
        
        
        
    def readVTK(self, filename, order='F'):

        """
        Utility function to read vtk volume. 

        Params:

            - inherited from class (check at the beginning of the class)
            - path: path where VTK file is located (str)
            - filename: VTK file name (str)

        Returns:

            - numpy_array: VTK file read and transformed into NumPy array
            - origin: origin coordinates (list)
            - spacing: image resolution (list)

        """

        reader = vtk.vtkStructuredPointsReader()

        reader.SetFileName(filename)

        reader.Update()

        image = reader.GetOutput()

        numpy_array = vtk_to_numpy(image.GetPointData().GetScalars())

        numpy_array = numpy_array.reshape(image.GetDimensions(),order='F')

        numpy_array = numpy_array.swapaxes(0,1)

        origin = list(image.GetOrigin())

        spacing = list(image.GetSpacing())

        return numpy_array, origin, spacing

        
    def array2vtk(self, array, filename, dest_path, origin = [0,0,0], spacing = [1,1,1]):
                
        """
        Convert array into .vtk file
        
        - Params:
            
            inherited class parameters (see description at beginning of the class)
            
            array: array to be converted into .vtk file (np array)
            
            filename: filename with which to save array as VTK file (str)
            
            origin: origin of coordinate system, by default (0,0,0) (list)
            
            spacing: spacing of coordinate system, by default (1,1,1) (list)
        
        """
          
        vtk_writer = vtk.vtkStructuredPointsWriter()
    
            
        # Check if destination folder exists
        
        #print('Checking if destination folder exists\n')
            
        isdir = os.path.isdir(dest_path)
            
        if not isdir:
            
            os.makedirs(dest_path)
            
            print('Non-existing destination path. Created\n')
        
        # Check if files already exist in destination folder
            
        exist = filename in os.listdir(dest_path)
        
        overwrite = 'y'
        
        if exist:
            
            overwrite = input("File is already in folder. Do you want to overwrite? [y/n]\n")
        
        if overwrite == 'y' or overwrite == 'Y':
                
            vtk_writer.SetFileName(dest_path + filename)
                
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


        
        

    def testing_dataLoader(self):

        """
        Data loader providing raw file and mask paths for model testing

        Params:

            - test_path: folder where to start looking for all files (str, in here included as class argument)


        Returns:

            - raw_paths: list with paths for raw files (list of str)

            - mask_paths: list with paths for mask files (list of str)

            - total_patients: list with testing patients in 2D models (list of str)

        """

        raw_paths = []

        mask_paths = []

        total_patients = []

        if params.three_D: # 2D+time files

            if not('ckd1' in self.test_path): # Cross-validation or testing datasets

                studies = sorted(os.listdir(self.test_path)) # Access study folders

                for study in studies:

                    if study != '_ckd1':

                        study_path = self.test_path + study + '/'

                        patients = sorted(os.listdir(study_path)) # Access patient folders from each study

                        for patient in patients:

                            image_path = study_path + patient + '/'

                            images = sorted(os.listdir(image_path)) # Access 2D+time images inside each patient

                            if not('BF' in params.train_with): # Decide on modality to read

                                if 'mag_' in params.train_with or 'both' in params.train_with: # Magnitude

                                    ind_raw = [i for i,s in enumerate(images) if 'mag_' in s]

                                elif 'pha' in params.train_with: # Phase

                                    ind_raw = [i for i,s in enumerate(images) if 'pha' in s]

                                ind_msk = [i for i,s in enumerate(images) if 'msk' in s] # Masks

                                for ind_r, ind_m in zip(ind_raw, ind_msk):

                                    raw_paths.append(image_path + images[ind_r]) # Final list of image paths

                                    mask_paths.append(image_path + images[ind_m]) # Final list of mask paths


                            else:

                                ind_raw = [i for i,s in enumerate(images) if 'magBF' in s]

                                ind_msk = [i for i,s in enumerate(images) if 'msk' in s]

                                for ind_r, ind_m in zip(ind_raw, ind_msk):

                                    raw_paths.append(image_path + images[ind_r]) # Final list of image paths

                                    mask_paths.append(image_path + images[ind_m]) # Final list of mask paths


            else: # CKD1 study

                patients = sorted(os.listdir(self.test_path))

                for patient in patients:

                    if not('CKD015' in patient): # CKD015 patient has wrong measurements, skip it

                        image_path = self.test_path + patient + '/'

                        images = sorted(os.listdir(image_path)) # Images of the same patient

                        if not('BF' in params.train_with): # Modalities to access

                            if 'mag_' in params.train_with or 'both' in params.train_with:

                                ind_raw = [i for i,s in enumerate(images) if 'mag_' in s]

                            elif 'pha' in params.train_with:

                                ind_raw = [i for i,s in enumerate(images) if 'pha' in s]

                            ind_msk = [i for i,s in enumerate(images) if 'msk' in s]

                            for ind_r, ind_m in zip(ind_raw, ind_msk):

                                raw_paths.append(image_path + images[ind_r]) # Final list of image paths

                                mask_paths.append(image_path + images[ind_m]) # Final list of mask paths


                        else:

                            ind_raw = [i for i,s in enumerate(images) if 'magBF' in s]

                            ind_msk = [i for i,s in enumerate(images) if 'msk' in s]

                            for ind_r, ind_m in zip(ind_raw, ind_msk):

                                raw_paths.append(image_path + images[ind_r]) # Final list of image paths

                                mask_paths.append(image_path + images[ind_m]) # Final list of mask paths



        else: # 2D models

            if not('ckd1' in test_path): # Cross-validation or testing

                studies = sorted(os.listdir(self.test_path)) # List of studies

                for study in studies:

                    if study != '_ckd1':

                        study_path = self.test_path + study + '/'

                        patients = sorted(os.listdir(study_path)) # Patients of same study

                        for patient in patients:

                            patient_path = study_path + patient + '/' 

                            modalities = sorted(os.listdir(patient_path)) # Modalities of same patient

                            for modality in modalities:

                                if modality == 'msk': # Mask

                                    modality_path = patient_path + modality + '/'

                                    masks = sorted(os.listdir(modality_path))

                                    for mask in masks:

                                        mask_paths.append(modality_path + mask)

                                if not('BF' in params.train_with):

                                    if modality == 'mag': # Magnitude. There is no code for phase or magnitude + phase, so write it if you need it

                                        modality_path = patient_path + modality + '/'

                                        mags = sorted(os.listdir(modality_path))

                                        found = 0

                                        for mag in mags:

                                            if not('sum' in mag) and not('mip' in mag): # Separate frames

                                                raw_paths.append(modality_path + mag)

                                                ind_frame = mag.index('_frame')

                                                total_patients.append(mag[:ind_frame])

                                else:

                                    if modality == 'magBF': # Magnitude corrected in BF (BF = bias field)

                                        modality_path = patient_path + modality + '/'

                                        mags = sorted(os.listdir(modality_path))

                                        found = 0

                                        for mag in mags:

                                            if not('sum' in mag) and not('mip' in mag):

                                                raw_paths.append(modality_path + mag)

                                                ind_frame = mag.index('_frame')

                                                total_patients.append(mag[:ind_frame])

                                total_patients = np.unique(np.array(total_patients)).tolist()

            else: # CKD1 study

                patients = sorted(os.listdir(self.test_path)) # CKD1 patients

                for patient in patients: 

                    if not('CKD015' in patient): # Skip this patient

                        patient_path = self.test_path + patient + '/'

                        modalities = sorted(os.listdir(patient_path)) # See available modalities

                        for modality in modalities:        

                            if not('BF' in params.train_with):

                                if modality == 'mag':

                                    modality_path = patient_path + modality + '/'

                                    mags = sorted(os.listdir(modality_path))

                                    for mag in mags:

                                        if not('sum' in mag) and not('mip' in mag):

                                            raw_paths.append(modality_path + mag)

                                            ind_frame = mag.index('_frame')

                                            total_patients.append(mag[:ind_frame])
                                            

                            else:

                                if modality == 'magBF':

                                    modality_path = patient_path + modality + '/'

                                    mags = sorted(os.listdir(modality_path))

                                    for mag in mags:

                                        if not('sum' in mag) and not('mip' in mag):

                                            raw_paths.append(modality_path + mag)

                                            ind_frame = mag.index('_frame')

                                            total_patients.append(mag[:ind_frame])

                            total_patients = np.unique(np.array(total_patients)).tolist()

        return raw_paths, mask_paths, total_patients



    def modelPreparation(self, model_filename):

        """
        Prepares model and optimizer to load them for inference.
        
        Params:
        
            - inherited by class (see description at the beginning of the class)
            
            - model_filename: str with .tar file with trained model
            
        Outputs:
        
            - net: loaded trained model (PyTorch model)
            
            - optimizer: loaded PyTorch optimizer (PyTorch optimizer)

        Returns: list with loaded models and optimizers

        """

        if params.architecture == "UNet_with_Residuals":

            net = models.UNet_with_Residuals().cuda()

        elif params.architecture == "UNet_with_ResidualsPretrained":

            net = UNet_with_ResidualsPretrained().cuda()


        elif params.architecture == "NewUNet_with_Residuals":

            net = models.NewUNet_with_Residuals().cuda() 


        elif params.architecture == "AttentionUNet":

            net = models.AttentionUNet().cuda()


        elif params.architecture == "NewAttentionUNet":

            net = models.NewAttentionUNet().cuda()
            
        elif params.architecture == "AttentionUNetAutocontext":
        
            net = models.AttentionUNetAutocontext().cuda()
            
        elif params.architecture == "TimeDistributedAttentionUNet":
        
            net = models.TimeDistributedAttentionUNet().cuda()
            
        elif params.architecture == "TimeDistributedUNet":
        
            net = models.TimeDistributedUNet().cuda()
            
        elif params.architecture == "TimeDistributedUNetAutocontext":
        
            net = models.TimeDistributedUNetAutocontext().cuda()
            
            
        elif params.architecture == "Hourglass":
        
            net = models.Hourglass().cuda()

            
        elif params.architecture == "TimeDistributedAttentionUNetAutocontext":
        
            net = models.TimeDistributedAttentionUNetAutocontext().cuda()


        else:

            print('Wrong architecture. Please introduce a valid architecture')


        if params.opt == 'Adam':

            optimizer = optim.Adam(net.parameters(), params.lr)

        elif params.opt == 'RMSprop':

            optimizer = optim.RMSprop(net.parameters(), params.lr)

        elif params.opt == 'SGD':

            optimizer = optim.SGD(net.parameters(), params.lr)
            
        # Other optimizers, if you want to include them

        else:

            print('\nWrong optimizer. Please define a valid optimizer (Adam/RMSprop/SGD)\n')



        net, optimizer = utilities.model_loading(net, optimizer, self.model_path, model_filename)

        return net, optimizer


    def atoi(self, text):

        """

        Function helping to sort a list of strings with numbers inside
        
        Params:
        
            - inherited by class (see description at the beginning of the class)
            
            - text: str, usually with some number inside, to be sorted (str)
            
        Outputs:
        
            - number inside str as int

        """

        return int(text) if text.isdigit() else text
    
    

    def natural_keys(self, text):

        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        
        '''
def atoi(self, text):

        """

        Function helping to sort a list of strings with numbers inside
        
        Params:
        
            - inherited by class (see description at the beginning of the class)
            
            - text: str, usually with some number inside, to be sorted (str)
            
        Outputs:
        
            - number inside str as int

        """

        return int(text) if text.isdigit() else text
    
    

    def natural_keys(self, text):

        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        
        '''

        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]

        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]



    def Segmentation(self, X, out, Y):

        """
        Provide a segmentation VTK file with testing results of mask overlapped
        with normal raw image. 
        
        (Now I only use it for generating a 2D+time volume with colors for segmentation results, being inputted to MIP function, but not for file saving)

        Params:

            - X: raw image array (array)

            - out: output tensor from neural network (binary array)

            - Y: mask array (if unavailable, None) (binary array)


        Results:

            saved VTK image in color (G: correct segmentation B: under segmentation
            R: over segmentation)


        """

        final_result = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3))

        for i in range(final_result.shape[-1]): # Gray --> RGB

            final_result[:,:,:,i] = (X-np.amin(X))/(np.amax(X)-np.amin(X)) # Grayscale values from 0 to 1

        if Y is None:

            # If no mask available as reference, leave segmented result in green

            ind_green = np.array(np.where(out > 0.5))

            final_result[ind_green[0,:], ind_green[1,:], ind_green[2,:], 0] = 0

            final_result[ind_green[0,:], ind_green[1,:], ind_green[2,:], 1] = 1

            final_result[ind_green[0,:], ind_green[1,:], ind_green[2,:], 2] = 0


        else:


            # If mask is available, leave true positives in green, false positives red and false negatives in blue

            ind_pos_out = np.where(out > 0) # Result locations

            ind_pos_Y = np.where(Y > 0) # Mask locations

            out_aux = np.copy(out)

            Y_aux = np.copy(Y)

            out_aux[ind_pos_out] = 2 # Set result locations to 2

            Y_aux[ind_pos_Y] = 3 # Set mask locations to 3

            mult = out*Y # Intersection of mask and result

            ind_green = np.array(np.where(mult > 0))

            total = out_aux + Y_aux

            ind_red = np.array(np.where(total == 2)) # False positives

            ind_blue = np.array(np.where(total == 3)) # False negatives

            # Color final mask result

            final_result[ind_red[0,:], ind_red[1,:], ind_red[2,:], 0]  = 1

            final_result[ind_red[0,:], ind_red[1,:], ind_red[2,:], 1]  = 0

            final_result[ind_red[0,:], ind_red[1,:], ind_red[2,:], 2]  = 0

            final_result[ind_green[0,:], ind_green[1,:], ind_green[2,:], 0] = 0

            final_result[ind_green[0,:], ind_green[1,:], ind_green[2,:], 1] = 1

            final_result[ind_green[0,:], ind_green[1,:], ind_green[2,:], 2] = 0

            final_result[ind_blue[0,:], ind_blue[1,:], ind_blue[2,:], 0] = 0

            final_result[ind_blue[0,:], ind_blue[1,:], ind_blue[2,:], 1] = 0

            final_result[ind_blue[0,:], ind_blue[1,:], ind_blue[2,:], 2] = 1


        return final_result


    def MIP(self, X, out, Y, img_filename):

        """
        Provide a MIP image with found result.

        Params:

            - X: raw image array (array)

            - out: output tensor from neural network (array)

            - mask: provided mask tensor (if unavailable: None) (array)

            - img_filename: filename ID with which to save MIP image as PNG (str)

            - dest_path: destination folder (array)


        Results:

            saved PNG image in color (G: correct segmentation B: under segmentation
            R: over segmentation)


        """

        final_result = self.Segmentation(X, out, Y) # 2D+time with color information (H,W,T,color)

        mip_result = np.round(np.sum(255*final_result, axis = 0)/final_result.shape[2]).astype(int) # Volume from 0 to 255

        # Decide on final filename

        if 'pha' in params.train_with:

            filename = img_filename.replace('pha','mip') + '.png'

        elif not('pha' in params.train_with) and not('BF' in params.train_with):

            filename = img_filename.replace('mag','mip') + '.png'

        elif not('pha' in params.train_with) and ('BF' in params.train_with):

            filename = img_filename.replace('magBF','mip') + '.png'

        # Exchange blue and red channels to write PNG file with CV2

        mip_aux = np.copy(mip_result)

        mip_aux[:,:,0] = mip_result[:,:,-1]

        mip_aux[:,:,-1] = mip_result[:,:,0]

        cv2.imwrite(self.dest_path + filename, mip_aux)


    
    def majorityVoting(self, results):

        """
        Implement a majority voting of the results given by the different models from the cross-validation folds

        Params:

            - inherited by the class

            - results: list with model results (list of binary arrays)

        Returns:

            - result: result with majority voting (binary array)

        """

        s = sum(results)
        
        result = (s > 0) # Originally I was doing majority voting by thresholding on half the maximum, but taking all results > 0 seemed to work better
        
        #result1 = (s == 1)
        
        #result2 = (s == len(results))
        
        #new_s = sum([result1, result2])
        
        #result = (new_s > 0)

        return result



    def flowFromMask(self, mask, raw, spacing):

        """
        Compute flow parameters from masks and phase images.

        Params:

            - inherited from class (check at the beginning of the class)
            - mask: binary or quasi-binary 3D array with results from neural network segmentation (array)
            - raw: corresponding velocity array, extracted from phase images (array)
            - spacing: pixel size, useful to extract area and flow information (list)

        Returns:

            - result: 2D array with results on average velocity, standard deviation,
                    maximum velocity, minimum velocity, area, net flow, positive flow
                    and negative flow (array)

        """

        mult = mask * raw # Focus only where mask = 1

        result = np.zeros((8, mask.shape[0]))

        for j in range(mask.shape[0]): # Frame by frame analysis

            mult_frame = mult[j,:,:] # Frame of the multiplication operator

            s = np.sum(mult_frame.flatten()) # Overall sum of velocities for each frame in the segmented ROI

            ind = np.where(mult_frame != 0) # Indexes inside the ROI
            
            if len(mult_frame[ind]) != 0:

                result[0,j] = np.mean(mult_frame[ind].flatten()) # Mean values (cm/s)

                result[1,j] = np.std(mult_frame[ind].flatten()) # Standard deviation (cm/s)

                result[2,j] = np.amax(np.abs(mult_frame[ind])) # Maximum value (cm/s)

                result[3,j] = np.amin(np.abs(mult_frame[ind])) # Minimum value (cm/s)
                
                result[4,j] = (len(ind))*spacing[0]*spacing[1] # Area (cm2)
                
                aux = mult_frame[ind]
                
                pos = np.where(aux > 0) # Positive flow indexes
                
                neg = np.where(aux < 0) # Negative flow indexes
                
                result[6,j] = np.sum(aux[pos].flatten())*spacing[0]*spacing[1]/100 # Positive flow

                result[7,j] = np.sum(aux[neg].flatten())*spacing[0]*spacing[1]/100 # Negative flow

                result[5,j] = result[6,j] + result[7,j] # Net flow


        return result


    def phase2velocity(self, phase_array, img_filename):

        """
        Transforms raw phase image into velocity map, to later extract flow information. (Unused)

        Params:

            - phase array: array with phase values (array)

        Returns:

            - vel_array: array with velocity values (array)


        """

        # Extract VENC info to leave gray values from -VENC to +VENC

        vencs = np.loadtxt('/home/andres/Documents/_Data/venc_info/' + 'venc_values.txt')

        names = np.loadtxt('/home/andres/Documents/_Data/venc_info/' + 'venc_files.txt', dtype = 'str')

        study = img_filename[:4]

        if 'BF' in params.train_with:

            name = img_filename.replace('magBF_','')

        elif 'mag_' in params.train_with or 'both' in params.train_with:

            name = img_filename.replace('mag_','')

        elif 'pha' in params.train_with:

            name = img_filename.replace('pha_','')


        if params.prep_step != 'raw':

            name = name.replace('_' + params.prep_step, '')


        ind = np.where(names == name)

        venc = vencs[ind]

        # Analyze CKD1 and CKD2 studies

        # Set phase from 0 to +1, being scaled by the VENC

        vel_array = phase_array/venc[0]

        vel_array = vel_array + 1

        vel_array = vel_array*0.5

        return vel_array, venc[0]


    def excelInfo(self, flow_path, img_filename):

        """
        Extract flow information from Excel file.
        
        Params:
        
            - self: inherited arguments by class (see description in beginning of class)
        
            - flow_path: folder where Excel file with flow measurements is located
            
            - img_filename: image whose flow has to be accessed


        Returns:
        
            - array with mean flow value from image in Excel file
            
            - array with minimum velocity from image in Excel file
            
            - array with maximum velocity from image in Excel file



        """

        # Extract measured patients from CKD1

        target_patient = img_filename[5:11] # Patient to look for in Excel file

        target_rep = img_filename[12:16] # Acquisition to look for in Excel file

        target_orient = img_filename[17:19] # Orientation (dx/sin) to look for in Excel file

        df = pd.read_excel(flow_path + self.excel_file) # can also index sheet by name or fetch all sheets

        flow = np.array(df['Mean_arterial_flow'].tolist()) # Array with flow values

        max_v = np.array(df['Peak_velocity_max'].tolist()) # Array with maximum velocity values

        min_v = np.array(df['Peak_velocity_min'].tolist()) # Array with minimum velocity flow values

        rep = np.array(df['MR_Visit']) # Array with repetitions

        patients = df['Subject_ID'].tolist() # List of patient IDs

        orient_ckd1 = np.array(df['left_right_kidney'].tolist())  # Array with orientations

        ind_left = np.where(orient_ckd1 == 'left')

        ind_right = np.where(orient_ckd1 == 'right')

        orient_final = np.copy(orient_ckd1) # List with kidney orientations, changed as "si" for left kidney and "dx" for right kidney

        orient_final[ind_left] = 'si'

        orient_final[ind_right] = 'dx' 

        # Patient measurements

        patient_ind = np.array([i for i, s in enumerate(patients) if target_patient in s]).astype(int) # Indexes of same patient

        orients = orient_final[patient_ind] # Patient orientations

        ind_orient = np.array([i for i, s in enumerate(orients) if target_orient in s]).astype(int)

        patient_orient_ind = patient_ind[ind_orient] # Indexes of patient with same kidney orientation

        patient_rep = rep[patient_orient_ind] # Patient visits with same orientation

        ind_rep = np.array([i for i, s in enumerate(patient_rep) if target_rep in s]).astype(int)

        patient_orient_rep_ind = patient_orient_ind[ind_rep][0]

        return flow[patient_orient_rep_ind], max_v[patient_orient_rep_ind], min_v[patient_orient_rep_ind]


    def timeExtractor(self, file_id):
        
        """
        Extract the corresponding time resolution for a given file. Unused
        
        Params:
        
            - inherited by the class
            
            - file_id: volumetric file identifier to look for in time (str)
            
        Returns:
        
            - time: found time resolution (float)
        
        """
        
        time_path = '/home/andres/Documents/_Data/Time_info/' # Change this if you change the location of this info

        time_files_file = 'time_info_files.txt'

        time_values_file = 'time_info_times.txt'

        time_files = np.loadtxt(time_path + time_files_file, dtype = str) # Array of files

        time_values = np.loadtxt(time_path + time_values_file) # Array of time values

        file_elements = file_id.split('_') 
        
        # Identify time file and time values of 2D+time file to look for

        if 'CKD' in file_elements[0]:

            file_to_look = file_elements[0] + '_' + file_elements[1] + '_' + file_elements[2] + '_' + file_elements[3] + '_' + file_elements[-2]

        elif file_elements[0] == 'Hero':

            file_elements[0] = 'hero'

            if 'venc' in file_elements[-2]:

                file_to_look = file_elements[0] + '_' + file_elements[1] + '_' + file_elements[-5] + '_' + file_elements[-3] + '_' + file_elements[-2]

            elif '20190416' in file_elements[1] or '20190521' in file_elements[1]:

                file_to_look = file_elements[0] + '_' + file_elements[1] + '_' + file_elements[2] + '_' + file_elements[-4] + '_' + file_elements[-2]
            
            else:
                
                file_to_look = file_elements[0] + '_' + file_elements[1] + '_' + file_elements[2] + '_' + file_elements[-1]
            
            
        elif file_elements[0] == 'Extr':

            file_elements[0] = 'extr'    

            file_to_look = file_elements[0] + '_' + file_elements[1] + '_' + file_elements[2] + '_' + file_elements[-4] + '_' + file_elements[-2]


        ind_file = np.where(time_files == file_to_look)

        time = time_values[ind_file][0] 
        
        return time
    
    
    def maskSaving(self, array, file_id, origin, spacing):
        
        """
        Save results from network as VTK files, to later read these files and save their slices as PNG files for Segment software visualization
        
        Params:
        
            - inherited by class (see description at the beginning)
        
            - array: network result (array)
            
            - file_id: 2D+time file ID to be searched (str)
            
            - origin: image origin coordinates (list)
            
            - spacing: image resolution (list)
        
        """
        
        
        if not('both' in params.train_with):
                    
            if not('mag_' in params.train_with):

                filename = file_id.replace(params.train_with, 'net')

            else:

                filename = file_id.replace(params.train_with, 'net_')

        else:

            if 'BF' in params.train_with:

                filename = file_id.replace('magBF', 'net')

            else:

                filename = file_id.replace('mag', 'net')

        filename += '.vtk'

        self.array2vtk(np.rollaxis(array, 0, -1), filename, self.dest_path, origin, spacing)
        
    
    def combineViews(self, list_views):
        
        """
        Combine network results from axial, coronal and sagittal views, through majority voting. Unused
        
        Params:
        
            - inherited by class (see description at the beginning)
        
            - list_views: list with axial, coronal and sagittal arrays (list of arrays)
        
        """
        
        axial = np.squeeze(list_views[0])
        
        coronal = np.squeeze(list_views[1])
        
        sagittal = np.squeeze(list_views[2])
        
        # Axes swapping or rolling for dimensions to coincide
        
        sagittal = np.swapaxes(sagittal, 0, -1)
        
        coronal = np.rollaxis(coronal, -1, 0)
        
        result = self.majorityVoting([axial, coronal, sagittal])
        
        return result
    
    
    def patientFiles(self, paths):
        
        """
        For cross-validation set evaluation, get an array of indexes indicating the network to be used for evaluating each case
        
        Params:
        
            - self.model_path: folder where to look for the files with the patient partitions
            
            - paths: list of paths with images to assign a certain network for evaluation (list of str)
            
        Returns:
        
            - final_indexes: array stating the network index to be used for evaluating one file (from several network folds inputted to the class, provide the index of that network fold where the file that is being evaluated was in the validation set during model training)
        
        """
        
        files = sorted(os.listdir(self.model_path))
        
        ind_files = [i for i,s in enumerate(files) if 'PatientPartioning' in s] # Check patients for training and validation in each fold accessing .txt files on partitioning generated during model training
        
        final_indexes = np.zeros(len(paths))
        
        cont = 0
        
        for i in ind_files:
            
            # Go through all partitionings
            
            found = 0
            
            patients = []
            
            with open(self.model_path + files[i]) as partition:
        
                for line in partition:
                
                    if found == 1:
                        
                        # Find all files from that patient in the images list
                        
                        ind_patient = [i for i,s in enumerate(sorted(paths)) if line[:-1] in s]
                        
                        final_indexes[ind_patient] = cont
                        
                        patients.append(line[:-1])
                
                    if 'validation' in line:
                        
                        found = 1
                        
            cont += 1
  
        return final_indexes
            
        
                    


    def __main__(self):

        t1 = time.time()

        # Load raw files and mask paths for testing

        r_paths, m_paths, file_ids = self.testing_dataLoader()
        
        params.test = True

        # Load dataset of interest

        dataset = QFlowDataset(r_paths, m_paths, False, False)

        test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = 0)
        
        if 'crop' in self.test_path: # This has just been done for the 'crop' preprocessing pipeline. Needs to be programmed for other preprocessing pipelines
            
            net_indexes = self.patientFiles(r_paths) # Array with net indexes indicating the network used for validating each image


        net_results = [] # List of network results

        result_flows = [] # List with flows computed from the networks
        
        axial_nets = [] # List with axial results from the networks
                
        coronal_nets = [] # List with coronal results from the networks

        sagittal_nets = [] # List with sagittal results from the networks

        axial_raw = [] # List with axial images

        coronal_raw = [] # List with coronal images

        sagittal_raw = [] # List with sagittal images

        axial_gt = [] # List with axial masks

        coronal_gt = [] # List with coronal masks

        sagittal_gt = [] # List with sagittal masks

        axial_name = [] # List with axial filenames

        coronal_name = [] # List with coronal filenames

        sagittal_name = [] # List with sagittal filenames

        # Evaluate results

        for i in range(len(self.model_filenames)): # Iterate through the models of each fold

            net, optimizer = self.modelPreparation(self.model_filenames[i]) # Model and optimizer loading

            res, raw, net_result, gt, names = evaluate.evaluate(net, test_loader, 0, 'test') # Evaluation of loaded files with loaded model
            
            if not(params.multi_view):

                net_results.append(np.squeeze(np.array(net_result))) # Result appending
                
            else:
                
                # Find indexes where names[-4] != 'l' --> axial slices
                
                axial_net = []
                
                coronal_net = []
                
                sagittal_net = []
                
                cont_name = 0
                
                for name in names:
                    
                    if name[-5] != 'l':
                        
                        if i == 0:
                        
                            axial_name.append(name)

                            axial_raw.append(raw[cont_name])

                            axial_gt.append(gt[cont_name])
                        
                        axial_net.append(np.squeeze(net_result[cont_name]))
                        
                    else:
                        
                        if 'sagittal' in name:
                            
                            if i == 0:
                            
                                sagittal_name.append(name)

                                sagittal_raw.append(raw[cont_name])

                                sagittal_gt.append(gt[cont_name])

                            sagittal_net.append(np.squeeze(net_result[cont_name]))
                            
                        elif 'coronal' in name:
                            
                            if i == 0:
                            
                                coronal_name.append(name)

                                coronal_raw.append(raw[cont_name])

                                coronal_gt.append(gt[cont_name])

                            coronal_net.append(np.squeeze(net_result[cont_name]))


                    cont_name += 1
                    
                if i == 0: # In first iteration, save names of axial, coronal and sagittal files
                    
                    axial_name_array = np.array(axial_name)
                    
                    coronal_name_array = np.array(coronal_name)
                    
                    sagittal_name_array = np.array(sagittal_name)
                    
                    axial_raw_array = np.array(axial_raw)
                    
                    axial_gt_array = np.array(axial_gt)
                    
                    axial_name.sort(key = self.natural_keys) # Sort according to slice number in filename

                    coronal_name.sort(key = self.natural_keys)

                    sagittal_name.sort(key = self.natural_keys)
                    
                axial_nets.append(axial_net)
                
                coronal_nets.append(coronal_net)
                
                sagittal_nets.append(sagittal_net)
                
                    

        names_array = np.array(names) # Array with filenames
        
        if not(params.multi_view): # Only one view case

            raw_array = np.squeeze(np.array(raw))

            if len(gt) != 0:

                gt_array = np.squeeze(np.array(gt))

        dices = [] # List with Dice coefficients computed for each 2D+time

        precisions = [] # List with precisions computed for each 2D+time

        recalls = [] # List with recalls computed for each 2D+time

        # Extract volumetric file IDs
        
        times = [] # List with time resolution per file

        for file_id in file_ids: # Iterate through all volumes found
            
            if not(params.multi_view): # One-view case
            
                #times.append(self.timeExtractor(file_id)) # List of times

                ind_file = [i for i,s in enumerate(names) if file_id in s] # Indexes of loaded images and masks belonging to the volume we are interested in

                names_file = names_array[ind_file].tolist() # Filenames of images in the same volume

                names_file.sort(key = self.natural_keys) # Sort according to slice number in filename

                inds = []

                pha_array = []
                
                for name in names_file: # Iterate through all filenames of the same image

                    ind = np.where(names_array == name)[0][0]

                    inds.append(ind) # Indexes of slices from same volume in the data loader

                    if not('pha' in params.train_with) and not('both' in params.train_with):

                        # Phase array obtention for flow computation, only if we train with phase images or with magnitude + phase images

                        if 'magBF' in params.train_with:

                            pha_path = r_paths[ind].replace('magBF','pha')

                        elif 'mag_' in params.train_with:

                            pha_path = r_paths[ind].replace('mag','pha')

                        else:

                            print('Wrong key introduced for training. Please introduce a valid key ("mag_", "magBF", "pha", "both", "bothBF")')


                        pha_arr, origin, spacing = self.readVTK(pha_path)

                        pha_array.append(np.flip(np.squeeze(pha_arr), axis = 0))
                        
                net_results_file = [] # List of all network results for an only volumetric file
                        
                for m in range(len(net_results)):

                    net_r_array = np.squeeze(np.array(net_results[m]))

                    net_r_array_file = net_r_array[inds]

                    net_results_file.append(net_r_array_file)
                
            else: # Multi-view testing case
                
                # Look for indexes of axial, coronal and sagittal files inside all loaded images and masks
                
                ind_axial_file = [i for i,s in enumerate(axial_name) if file_id in s]
                
                ind_coronal_file = [i for i,s in enumerate(coronal_name) if file_id in s]
                
                ind_sagittal_file = [i for i,s in enumerate(sagittal_name) if file_id in s]
                
                # Filename sorting
                
                sort_axial_name_array = np.array(axial_name)
                
                sort_coronal_name_array = np.array(coronal_name)
                
                sort_sagittal_name_array = np.array(sagittal_name)
                
                # Filenames of images in axial, coronal or sagittal that belong to the same volume
                
                axial_name_file = sort_axial_name_array[ind_axial_file]
                
                coronal_name_file = sort_coronal_name_array[ind_coronal_file]
                
                sagittal_name_file = sort_sagittal_name_array[ind_sagittal_file]
                
                axial_inds = []
                
                coronal_inds = []
                
                sagittal_inds = []
                
                pha_array = []
                
                for axial_n in axial_name_file: # Load all axial files inside a volume
                    
                    ind = np.where(axial_name_array == axial_n)
                    
                    axial_inds.append(ind) # Indexes of slices from same volume in the data loader

                    if not('pha' in params.train_with) and not('both' in params.train_with):

                        # Phase array obtention for flow computation, only necessary in the axial view

                        if 'magBF' in params.train_with:

                            pha_path = str(axial_name_array[ind]).replace('magBF','pha')

                        elif 'mag_' in params.train_with:

                            pha_path = str(axial_name_array[ind]).replace('mag','pha')

                        else:

                            print('Wrong key introduced for training. Please introduce a valid key ("mag_", "magBF", "pha", "both", "bothBF")')
                        
                        
                        pha_arr, origin, spacing = self.readVTK(str(pha_path[2:-2]))

                        pha_array.append(np.flip(np.squeeze(pha_arr), axis = 0))
                        
                pha_array = np.squeeze(np.array(pha_array))
                        
                for coronal_n in coronal_name_file: # Get results from all coronal files within the same volume
                    
                    ind = np.where(coronal_name_array == coronal_n)
                    
                    coronal_inds.append(ind) # Indexes of slices from same volume in the data loader
                    
                for sagittal_n in sagittal_name_file: # Get results from all sagittal files within the same volume
                    
                    ind = np.where(sagittal_name_array == sagittal_n)
                    
                    sagittal_inds.append(ind) # Indexes of slices from same volume in the data loader 
                    
                raw_array_file = axial_raw_array[axial_inds] # Get images from same volume just from the axial view
                
                raw_file = np.squeeze(raw_array_file) # Magnitude file
                
                mag_file = raw_file[:,0,:,:] 
                
                net_axial_results_file = [] # List of axial network results inside the same volume
                
                net_coronal_results_file = [] # List of coronal network results inside the same volume
                
                net_sagittal_results_file = [] # List of sagittal network results inside the same volume

            
                # Extraction of network predictions from all cross-validation folds
                
                net_results_file = []

                for m in range(len(axial_nets)): 

                    net_r_array_axial = np.array(axial_nets[m]) 

                    net_r_array_axial_file = net_r_array_axial[axial_inds] # Axial network predictions from the same volume
                    
                    
                    net_r_array_coronal = np.array(coronal_nets[m])

                    net_r_array_coronal_file = net_r_array_coronal[coronal_inds] # Coronal network predictions from the same volume
                    
                    
                    net_r_array_sagittal = np.array(sagittal_nets[m])

                    net_r_array_sagittal_file = net_r_array_sagittal[sagittal_inds] # Sagittal network predictions from the same volume
                    
                                
                    # Combine results from all views
                
                    net_results_file.append(self.combineViews([net_r_array_axial_file, net_r_array_coronal_file, net_r_array_sagittal_file]))
      
            # Provide a unique prediction based on majority voting
        
            if not('crop' in self.test_path): # File evaluation for test set or for CKD1 set --> combine through majority voting the outputs of the models coming from different folds

                net_array_file = self.majorityVoting(net_results_file)
                
            else: # File evaluation for cross-validation images
                
                # From the different model outputs of the same volume, take only those outputs corresponding to those models where the volume was in the validation set
                
                f = 0
                
                c = 0
                
                while f == 0:
                    
                    if file_id in r_paths[c]:
                        
                        f = 1
                        
                    c += 1
                
                ind = int(net_indexes[c - 1])
                
                net_array_file = net_results_file[ind]
                
                
            
            # Post-processing on connected components: remove small peripheral components
            
            net_array_file = utilities.connectedComponentsPostProcessing(net_array_file)
            
            #self.maskSaving(net_array_file, file_id, origin, spacing) # Result saving as VTK files in a given folder (deactivated now, only useful for outputting results into Segment)
            
            if not(params.multi_view): # One-view

                raw_array_file = raw_array[inds]
                
                if len(raw_array_file.shape) == 4:

                    mag_file = raw_array_file[:,0,:,:] # Corresponding magnitude image from the volume analyzed 
                    
                else:
                    
                    mag_file = np.copy(raw_array_file)
                
                

            if len(gt) != 0: # In testing or cross-validation sets with masks, get corresponding masks to the volume analyzed
                
                if not(params.multi_view):

                    gt_array_file = gt_array[inds]
                    
                else:
                    
                    gt_array_file = np.squeeze(axial_gt_array[axial_inds])

                # Metric computation: network results vs masks (functions in utilities.py)

                dices.append(utilities.dice_coef(net_array_file,gt_array_file))

                precisions.append(utilities.precision(net_array_file,gt_array_file))

                recalls.append(utilities.recall(net_array_file,gt_array_file))

                # Obtain MIPs as PNG files in the given folder for saving

                self.MIP(mag_file, net_array_file, gt_array_file, file_id)

            else: # If there are no masks (CKD1 case), compute only MIP files

                self.MIP(mag_file, net_array_file, None, file_id)  # Obtain MIPs as PNG files in the given folder for saving
                
                
            net_array_file = utilities.outputDilation(net_array_file) # Dilate resulting masks for a more accurate flow computation


            # Obtain corresponding phase volume along time for flow extraction. Phase volumes contain blood velocities

            if 'pha' in params.train_with:
                
                pha_array = np.copy(mag_file) # If training with phase, just access information from the input arrays to the network

            elif 'both' in params.train_with: # If training with magnitude + phase, just access information from the input arrays to the network (channel 3 if projections are being inputted or 1 if no projection is inputted)


                if params.sum_work:

                    pha_array = raw_array_file[:,3,:,:]

                else:

                    pha_array = raw_array_file[:,1,:,:]

            pha_array = np.array(pha_array)

            flow_result = self.flowFromMask(net_array_file, pha_array, spacing) # Reference flow computation from results

            if len(gt) == 0: # If there are no masks available (CKD1 set)

                mean_flow = np.mean(flow_result[5,:])

                min_v = np.min(np.abs(flow_result[0,:]))

                max_v = np.max(np.abs(flow_result[0,:]))

                mean_flow_gt, max_v_gt, min_v_gt = self.excelInfo(self.flow_paths[0], file_id) # Extract reference mean flow, minimum and maximum velocity in CKD1 study from Excel file

                result_flows.append([abs(mean_flow), abs(mean_flow_gt)]) # Include result and reference flows for later comparison in this list
                
                # Bar plots comparing CKD1 flows (only one flow measurement for all frames). Saved into destination folder

                fig = plt.figure(figsize = (13,5))

                plt.bar(np.arange(2), [abs(mean_flow_gt), abs(mean_flow)], color = 'b')

                plt.title('Net flow comparison for ' + file_id)

                plt.xlabel('Reference and result flows')

                plt.ylabel('Flow (ml/s)')

                fig.savefig(self.dest_path + 'flow_comparison_' + file_id + '.png')


                fig = plt.figure(figsize = (13,5))

                plt.bar(np.arange(2), [abs(max_v_gt), abs(max_v)], color = 'b')

                plt.title('v_max comparison for ' + file_id)

                plt.xlabel('Reference and result v_max')

                plt.ylabel('v_max (cm/s)')

                fig.savefig(self.dest_path + 'v_max_comparison_' + file_id + '.png')


                fig = plt.figure(figsize = (13,5))

                plt.bar(np.arange(2), [abs(min_v_gt), abs(min_v)], color = 'b')

                plt.title('v_min comparison for ' + file_id)

                plt.xlabel('Reference and result v_min')

                plt.ylabel('v_min (cm/s)')

                fig.savefig(self.dest_path + 'v_min_comparison_' + file_id + '.png')

            else: # Masks available. Cross-validation or testing
                
                flow_result_mask = self.flowFromMask(gt_array_file, pha_array, spacing) # Flows from masks (alternative to be computed instead of importing flow measurements from Segment)

                # Extract flow information from txt_file. Access a different folder depending on the study

                # Search corresponding txt file

                if 'CKD2' in file_id:

                    flow_path = self.flow_paths[1]

                elif 'Hero' in file_id or 'hero' in file_id:

                    flow_path = self.flow_paths[2]

                elif 'Extr' in file_id or 'extr' in file_id:

                    flow_path = self.flow_paths[-1]

                txt_files = sorted(os.listdir(flow_path)) # All TXT files with flow information. Now search the corresponding to our volume. Look for that file that has the same patient, repetition number and orientation as the one we are looking for

                # Get index of corresponding txt file

                # Get orientation and repetition number of file

                if 'dx' in file_id:

                    orient = 'dx'


                elif 'si' in file_id:

                    orient = 'si'



                if '_0' in file_id:

                    rep = '_0'


                elif '_1' in file_id:

                    rep = '_1'


                ind = file_id.index(orient)


                ind_txt = [j for j, s in enumerate(txt_files) if file_id[:ind + 2] in s]

                ind_final = -1


                for k in ind_txt:

                    if not('venc' in file_id): # Files without VENC information in their names

                        if rep in txt_files[k]:

                            ind_final = k

                    else: # Files with VENC information in their names 

                        if ('venc080' in txt_files[k] and 'venc080' in file_id) and (rep in txt_files[k]):

                            ind_final = k

                        elif ('venc100' in txt_files[k] and 'venc100' in file_id) and (rep in txt_files[k]):

                            ind_final = k

                        elif ('venc120' in txt_files[k] and 'venc120' in file_id) and (rep in txt_files[k]):

                            ind_final = k


                if k == -1:

                    print('Flow file not found')

                    exit()
                
                # Extract flow curves from result of network

                load_info_mat = FlowInfo(file_id[:4], flow_path, None, None, 'load', True, txt_files[ind_final]) # Array with information extracted from TXT files

                mean_v, std_v, max_v, min_v, energy, area, net_flow, pos_flow, neg_flow = load_info_mat.__main__() # Flow variables extracted from TXT files
                
                # Plot of reference flows vs resulting flows. Information is now time-resolved in different frames. Saved in destination folder
                
                fig = plt.figure(figsize = (13,5))

                plt.plot(np.arange(1, flow_result.shape[1] + 1), np.abs(net_flow), 'b', label = 'External software')

                plt.plot(np.arange(1, flow_result.shape[1] + 1), np.abs(flow_result[5,:]), 'r', label = 'Result')

                #plt.plot(np.arange(1, flow_result.shape[1] + 1), np.abs(flow_result_mask[5,:]), 'g', label = 'Ground truth')
                
                plt.title('Flow comparison')

                plt.xlabel('Time frame #')

                plt.ylabel('Flow (ml/s)')

                plt.legend()

                fig.savefig(dest_path + 'flow_comparison_' + file_id + '.png')

                result_flows.append([np.abs(flow_result[5,:]), np.abs(flow_result_mask[5,:]), np.abs(net_flow)])        


        result_metrics = []
        
        if len(gt) != 0: # If there are masks (cross-validation, testing), print metrics as mean +- std

            print('\nDice coefficient: {} +- {}\n'.format(np.mean(dices), np.std(dices)))

            print('Precision: {} +- {}\n'.format(np.mean(precisions), np.std(precisions)))

            print('Recall: {} +- {}\n'.format(np.mean(recalls), np.std(recalls)))
            
            result_metrics.append([dices, precisions, recalls])


        t2 = time.time()
        
        params.test = False

        print('\nMean testing time per volume: {} seconds\n'.format((t2-t1)/len(file_ids)))

        return result_flows, result_metrics, file_ids
    
    
    
    
def statisticsResults(result, reference, key, dest_path):
    
    
    """
    Import resulting and reference flows after inference testing for a statistical analysis. This function is for flow computation and is separated from the main testing class
    
    Params:
    
        - result: resulting flows (array)
        
        - reference: reference flows (array)
        
        - key: flag stating which kind of data is being analyzed (str)
        
        - dest_path: destination folder where to save statistics comparison results (str)
    
    """

    t_stat, p_value_t = flowStatistics.t_test(result, reference)  # t-test  

    s, p_value_w = flowStatistics.wilcoxon_test(result, reference) # Wilcoxon text
    
    coef, mse, r2, dist = flowStatistics.linear_regression_test(result, reference, True, True, dest_path, filename = 'regression_plot_2_' + key + '_' + params.architecture + '_' + params.train_with + '.png') # Linear regression + file saving in destination folder with plot

    flowStatistics.bland_altman_plot(result, reference, True, dest_path, filename = 'bland_altman_plot_' + key + '_' + params.architecture + '_' + params.train_with + '.png') # Bland-Altman plot + file saving in destination folder with plot

    r = flowStatistics.correlation(result, reference) # Pearson correlation (r)
                                     
    print('\nPearson Correlation Coefficient: {}\n'.format(r))
    
    # Write all statistical results in a TXT file saved in the destination folder
                                     
    with open(dest_path + 'Statistics_' + key + '_' + params.architecture + '_' + params.train_with + '.txt', 'w') as file:
                                     
        file.write('Pearson correlation coefficient: ' + str(r) + '\n')
                                     
        file.write('r2: ' + str(r2) + '\n')
                                     
        file.write('t statistic: ' + str(t_stat) + '\n')
                                     
        file.write('t test p value: ' + str(p_value_t) + '\n')
                                     
        file.write('Wilcoxon sum: ' + str(s) + '\n')
                                     
        file.write('Wilcoxon test p value: ' + str(p_value_w) + '\n')
                                     
        file.write('Mean squared error in linear regression: ' + str(mse) + '\n')
                                     
        file.write('Line equation: y = ' + str(coef[0]) + 'x + ' + str(coef[1]) + '\n')
        
        file.write('Result distribution: ' + str(dist[0]) + '+-' + str(dist[1]) + ' \\ Reference distribution: ' + str(dist[-2]) + '+-' + str(dist[-1]) + '\n')
    
    
# Use example:

# Folders with reference flow information from Segment

#flow_paths = ['/home/andres/Documents/_Data/CKD_Part1/', '/home/andres/Documents/_Data/CKD_Part2/4_Flow/',
 #             '/home/andres/Documents/_Data/Heroic/_Flow/', '/home/andres/Documents/_Data/Extra/_Flow/']


#venc_path = '/home/andres/Documents/_Data/venc_info/' # Folder with VENC information

#if params.three_D: (select one or the other)
    
    #test_path = '/home/andres/Documents/_Data/_Patients_Test/' # Path with testing images, for full 2D+time networks
    
    #test_path = '/home/andres/Documents/_Data/_Patients2D/_Pre_crop/' # Path with cross-validation images, for full 2D+time networks, in the "crop" pre-processing pipeline
    
#else:

    #test_path = '/home/andres/Documents/_Data/_Patients_Test2D/' # Path with testing images, for 2D models or for 2D+time networks trained only with a few neighboring slices
    
    #test_path = '/home/andres/Documents/_Data/_Patients2D/_Pre_crop/' # Path with cross-validation images, for 2D models or for 2D+time networks trained only with a few neighboring slices, in the "crop" pipeline
    
    
    
#if params.three_D:
    
    #test_path = '/home/andres/Documents/_Data/_Patients/_Pre_crop/_ckd1/' # Path with CKD1 images, for full 2D+time networks
    
#else:

 #   test_path = '/home/andres/Documents/_Data/_Patients2D/_Pre_crop/_ckd1/' # Path with CKD1 images, for 2D networks or 2D+time networks trained only with few neighboring frames, in the "crop" pipeline     


# Look for the needed files

#mod = ['trainedWithmagBF_cropfold_0.tar', 'trainedWithmagBF_cropfold_1.tar', 'trainedWithmagBF_cropfold_2.tar',
 #     'trainedWithmagBF_cropfold_3.tar'] # List of models for the different folds obtained (trained with magnitude images corrected in Bias Fields, with the "crop" pipeline)
    
#excel_file = 'CKD_QFlow_results.xlsx' # Excel file with flow measurements


#dest_path = '/home/andres/Documents/_Results_final/AttentionUNet_2DextraSupervision2iter_magBF/' # Path where to save results (usually somewhere in a Result network)

#model_path = '/home/andres/Documents/_Data/Network_data_new/AttentionUNetAutocontext_2DextraSupervision2iter/' # Path where the network files have been saved

#test = testing(test_path, mod, model_path, flow_paths, excel_file, dest_path) # Class call

#results, metric_results, files = test.__main__() # Execution

# Removal of one outlier that was a pain in the ass ('Hero_20190521_002_si_msk_0')

# Outlier removal 

#files_array = np.array(files) # Array with volumetric file IDs

#ind_remove = np.where(files_array == 'Hero_20190521_002_si_magBF_0_crop')
                
#results_array = np.array(results) # Array with reference and computed results

# Flow statistics computations

#if not('_ckd1' in test_path): # Cross-validation/testing

 #   res = [] # List with computed results

 #   ref = [] # List with reference results

 #   mean_res = [] # List with mean values for results, in all time frames

 #   rari_res = [] # List with RARI results: (max-min)/max

  #  max_res = [] # List with peak results

  #  mean_ref = [] # List with mean values for reference flows, in all time frames

   # max_ref = [] # List with peak references

   # rari_ref = [] # List with RARI references
    
  #  cont = 0

  #  for r in results: # Iterate results for a 2D+time volume
        
  #      if cont != ind_remove[0][0]: 

  #          res.append(r[0].tolist()) # List with all frame results

  #          ref.append(r[2].tolist()) # List with all frame references

  #          mean_res.append(np.mean(r[0]))

  #          mean_ref.append(np.mean(r[2]))

  #          max_res.append(np.max(r[0]))

  #          max_ref.append(np.max(r[2]))

  #          rari_res.append((np.max(r[0]) - np.min(r[0]))/np.max(r[0]))

  #          rari_ref.append((np.max(r[2]) - np.min(r[2]))/np.max(r[2]))

  #          cont += 1


  #  Transformation of lists into arrays  
    
  #  mean_res = np.array(mean_res)

  #  rari_res = np.array(rari_res)

  #  max_res = np.array(max_res)

  #  mean_ref = np.array(mean_ref)

  #  max_ref = np.array(max_ref)

  #  rari_ref = np.array(rari_ref)

  #  res = np.array(list(itertools.chain.from_iterable(res)))

  #  ref = np.array(list(itertools.chain.from_iterable(ref)))
    
  #  if 'Test' in test_path: # Key for filename, identifying if flows compared come from testing or from cross-validation

        # filename_key = 'test'
    
  #  else:

        # filename_key = 'cv'
    
  #  statisticsResults(mean_res, mean_ref, filename_key + '_means', dest_path) # Mean value comparison

  #  statisticsResults(res, ref, filename_key + '_samples', dest_path) # All frames comparison
    
  #  statisticsResults(rari_res, rari_ref, filename_key + '_rari', dest_path) # RARI comparison
    
  #  statisticsResults(max_res, max_ref, filename_key + '_max', dest_path) # Max value comparison

#else: # CKD1 set. Only able to compare mean flow results

  #  statisticsResults(results_array[:,0], results_array[:,1], 'ckd1', dest_path)  # Mean flow result comparison

