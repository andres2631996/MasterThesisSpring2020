#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:32:46 2020

@author: andres
"""

import numpy as np

import scipy.io

import os

import vtk

import matplotlib.pyplot as plt

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


# Code to extract flow information from ROIs in MAT files

# Compare results when I have segmentations from neural networks

# Export flow info in TXT files



# Extract results from neural network masks


class FlowInfo:
    
    """
    Extract or load flow information from MAT files or neural networks' masks.
    
    Params:
        
        - study: study where files belong to (str: 'CKD1'/'CKD2'/'Hero'/'Extr')
        
        - path: path of mask or MAT file for saving, path of TXT file for loading (str)
        
        - dest_path: path of TXT file for saving, None for loading (str, only used in 'load' mode)
        
        - raw_file_path: path of raw file. Only for saving from mask information. Otherwise is None (str, only used in 'save' mode)
        
        - flag: either 'save' for saving or 'load' for loading (str, states if 'save' mode or 'load' mode)
        
        - energy: in case of flow info loading, save the energy if available (bool, only in 'save' mode)
        
        - load_file: TXT file to read in case of file loading (str, only in 'load' mode)
    
    Returns:
        
        - if flag is 'save', saves flow information as TXT in location specified (dest_path)
        
        - if flag is 'load', loads flow information from TXT and provides arrays with flow information (used in test.py)
    
    """
    
    def __init__(self, study, path, dest_path, raw_file_path, flag, energy, file):
        
        self.study = study
        
        self.path = path
        
        self.dest_path = dest_path
        
        self.raw_file_path = raw_file_path

        self.flag = flag
        
        self.energy = energy
        
        self.load_file = file
        

    def readVTK(self, path, filename, order='F'):
            
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
    
        reader.SetFileName(path + filename)
    
        reader.Update()
    
        image = reader.GetOutput()
    
        numpy_array = vtk_to_numpy(image.GetPointData().GetScalars())
    
        numpy_array = numpy_array.reshape(image.GetDimensions(),order='F')
    
        numpy_array = numpy_array.swapaxes(0,1)
    
        origin = list(image.GetOrigin())
    
        spacing = list(image.GetSpacing())
    
        return numpy_array, origin, spacing



    def flowFromTXT(self):
        
        """
        Gets numpy array from TXT file with flow information.
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            - file: TXT filename (str)
        
        Returns:
            
            - flow parameters in array (array)
        
        """
    
        
        array = np.loadtxt(self.path + self.load_file)
        
        if self.energy:
            
            mean_v = array[0,:]
        
            std_v = array[1,:]
            
            max_v = array[2,:]
            
            min_v = array[3,:]
            
            energy = array[4,:]
            
            area = array[5,:]
        
            net_flow = array[6,:]
            
            pos_flow = array[7,:]
            
            neg_flow = array[8,:]
            
            return mean_v, std_v, max_v, min_v, energy, area, net_flow, pos_flow, neg_flow
        
        else:
        
            mean_v = array[0,:]
            
            std_v = array[1,:]
            
            max_v = array[2,:]
            
            min_v = array[3,:]
    
            area = array[4,:]
            
            net_flow = array[5,:]
            
            pos_flow = array[6,:]
            
            neg_flow = array[7,:]
            
            return mean_v, std_v, max_v, min_v, area, net_flow, pos_flow, neg_flow
    

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
    


    def mat_loader(self, filename):
            
        """
        Load .mat file from Segment software
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            
            - filename: name of MAT file to load, including its path (str)
        
        Returns:
            
            - mat: dictionary with .mat file fields (dictionary)
            
            - st: structure field with data (dictionary)
            
        """
        
        #print('Conversion from .mat into array\n')
        
        mat = scipy.io.loadmat(self.path + filename) 
    
        st = mat.get('setstruct') 
        
        return mat, st



    def flowFromMAT(self, filename):
        
        """
        Get flow information from measurements saved in MAT files. 
        
        These measurements are considered to be the ground truth measurements for flow by software
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            
            - filename: MAT filename (str)
            
        Returns:
        
            - array with flow information from the MAT file
        
        """

            
        mat, st = self.mat_loader(filename)
        
        img = st[0,0][0]

        res = np.zeros((10, img.shape[-1]))

        for j in range(len(st[0,0][53][0,0][7][0])):
            
            # See if there is flow information from more than one ROI
            
            cont = 0
        
            cont_correct = 0

            while cont_correct < (res.shape[0] - 1):

                if st[0,0][53][0,0][7][0,j][cont].shape[1] == res.shape[1]:
                    
                    # Flow information is located in the 53rd field of the dictionary encoding this file

                    res[cont_correct,:] += np.reshape(st[0,0][53][0,0][7][0,j][cont], res[cont,:].shape)
                    
                    if cont_correct != res.shape[1] - 1:
                    
                        cont_correct += 1
                    
                cont += 1

            return res[:-1,:]
    


    def __main__(self):
        
        if (self.flag == 'save') or (self.flag == 'Save') or (self.flag == 'SAVE'):
            
            # Save flow info from VTK masks or MAT files

            if self.raw_file_path is not None: # Save info from VTK masks
                
                raw_files = os.listdir(self.raw_file_path)
                
                # Load phase images' information. Look for corresponding phase image to the given VTK file mask
                
                raw_info = []

                for raw_file in raw_files:
                    
                    raw_info.append(raw_file[0:20] + raw_file[24:25]) # List with info from raw files
                
                raw_info_array = np.array(raw_info) # List into array
                
                # Load mask information

                mask_files = os.listdir(self.path)
                
                mask_info = []
                
                for mask_file in mask_files:

                    mask, _, _ = self.readVTK(self.path, mask_file) # VTK mask loading
                
                    mask_info = mask_file[0:20] + mask_file[24:25]
                    
                    ind = np.where(raw_info_array == mask_info)
                
                    for i in ind:
                        
                        raw, origin, spacing = self.readVTK(self.raw_file_path, raw_files[i])
                                        
                        res = self.flowFromMask(mask, raw, spacing)
                
                        # Save flow information as txt
        
                        np.savetxt(self.dest_path + raw_info[i] + '_flowInfo.txt', res)
                        
                
            
            else: # Save info from MAT files
                
                mat_files = sorted(os.listdir(self.path))
                
                for mat_file in mat_files:
    
                    if mat_file[-3:] == 'mat': # Access only MAT files
                        
                        res = self.flowFromMAT(mat_file)
                        
                        if self.study == 'CKD2': # CKD2 study
                            
                            ind_aux = mat_file.index('^')

                            filename_save = self.dest_path + self.study + '_' + mat_file[:ind_aux] + '_' + mat_file[ind_aux + 1: ind_aux + 5] + mat_file[-9:-4] + '_flowInfo.txt'

                            np.savetxt(filename_save, res)
                            
                        elif self.study == 'Hero': # Heroic study

                            filename_save = self.dest_path + mat_file[:-4].replace('msk_','') + '_flowInfo.txt'

                            np.savetxt(filename_save, res)
                            
                        elif self.study == 'Extr': # Extra study
                            
                            if '20181213' in mat_file: # Same flow measurement for two different acquisitions in that patient ID
                                
                                times = 2
                                
                            else:
                                
                                times = 1
                                
                            if 'sin' in mat_file or 'SI' in mat_file:
                                
                                orient = 'si'
                                
                            elif 'dx' in mat_file or 'DX' in mat_file:
                     
                                orient = 'dx'

                                
                            for time in range(times):

                                filename_save = self.study + '_' + mat_file[:12] + '_' + orient + '_' + str(time) + '_flowInfo.txt'

                                np.savetxt(self.dest_path + filename_save, res)
                        
                        
        
        elif (self.flag == 'load') or (self.flag == 'Load') or (self.flag == 'LOAD'):
            
            # Load flow info from TXT files that have been previously saved

            if self.energy: # Load flow energy information if it is available
                
                mean_v, std_v, max_v, min_v, energy, area, net_flow, pos_flow, neg_flow = self.flowFromTXT() 
                
                return mean_v, std_v, max_v, min_v, energy, area, net_flow, pos_flow, neg_flow
                
            else: # Unavailable flow energy information, do not load it
                
                mean_v, std_v, max_v, min_v, area, net_flow, pos_flow, neg_flow = self.flowFromTXT() 
                
                return mean_v, std_v, max_v, min_v, area, net_flow, pos_flow, neg_flow
            
            
 


                       
# Some test code...

   
# Get directory of phase images and different phase images
            
#study = 'Extr'
#
#raw_path = '/home/andres/Documents/_Data/CKD_Part2/3_Clean_Data/_pha/'
#
#mask_path = '/home/andres/Documents/_Data/Extra/_Binary_masks/'
#
#save_path = '/home/andres/Documents/_Data/Extra/_Flow/'
#
#gt_path = '/home/andres/Documents/_Data/Extra/_Test_measurements_Andres/' # Folder for ground truths
#
##txt_files = os.listdir(save_path)
#
## For saving info from VTK masks:
#
## save_info_mask = FlowInfo(mask_path, save_path, raw_path, 'save', None)
#
## For saving info from MAT files:
#
#save_info_mat = FlowInfo(study, gt_path, save_path, None, 'save', None, None)

# For loading info from VTK masks that has been previously saved as TXT:

# load_info_mask = FlowInfo(save_path, None, None, 'load', None)

# For saving info from MAT files that has been previously saved as TXT:

#load_info_mat = FlowInfo(save_path, None, None, 'load', True, txt_files[0])

# save_info_mask.__main__()

#save_info_mat.__main__()

# mean_v, std_v, max_v, min_v, area, net_flow, pos_flow, neg_flow = load_info_mask.__main__()

#mean_v, std_v, max_v, min_v, energy, area, net_flow, pos_flow, neg_flow = load_info_mat.__main__()

#plt.figure()

#plt.plot(np.arange(net_flow.shape[0]), net_flow)


# To run in terminal:

# mask_path = sys.argv[1]
# save_path = sys.argv[2] 
# raw_path = sys.argv[3]
# flag = sys.argv[4] 
# energy = sys.argv[5]
    


# if __name__ == "__main__":  