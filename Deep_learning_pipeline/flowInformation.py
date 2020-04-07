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
    Extract or load flow information from MAT files or masks.
    
    Params:
        
        - study: study where files belong to
        
        - path: path of mask or MAT file for saving, path of TXT file for loading
        
        - dest_path: path of TXT file for saving, None for loading
        
        - raw_file_path: path of raw file. Only for saving from mask information. Otherwise is None
        
        - flag: either 'save' for saving or 'load' for loading
        
        - energy: in case of flow info loading, save the energy if available
        
        - load_file: TXT file to read in case of file loading 
    
    Returns:
        
        - if flag is 'save', saves flow information as TXT in location specified
        
        - if flag is 'load', loads flow information from TXT and provides arrays
    
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
            - file: TXT file
        
        Returns:
            
            - flow parameters
        
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
            - mask: binary or quasi-binary 3D array with results from neural network segmentation
            - raw: corresponding phase image
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
            
            s = np.sum(mult_frame.flatten()) # Overall sum of each frame. If > 0 --> left kidney, if < 0 --> right kidney
            
            ind = np.where(mult_frame != 0) # Indexes inside the ROI
            
            result[0,j] = np.mean(mult_frame[ind].flatten()) # Mean values
            
            result[1,j] = np.std(mult_frame[ind].flatten()) # Standard deviation
            
            result[2,j] = np.amax(mult_frame) # Maximum value
            
            result[3,j] = np.amin(mult_frame) # Minimum value
            
            result[4,j] = (np.array(ind).shape[-1])*spacing[0]*spacing[1]/100 # Area
    
            result[5,j] = result[0,j]/result[4,j] # Net flow
            
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



    def mat_loader(self, filename):
            
        """
        Load .mat file from Segment software
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            
            - path and filename, inherited from the class
        
        Returns:
            
            - mat: dictionary with .mat file fields
            
            - st: structure field with data
            
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
            
            - path: folder with MAT files of interest
            
            - filename: MAT filename
        
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
                
                # Load phase images' information
                
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
                
                        ##       PERHAPS SOME SCALING PARAMETER NEEDS TO BE INTRODUCED!!!!
#
                        res = self.flowFromMask(mask, raw, spacing)
                
                        # Save flow information as txt
        
                        np.savetxt(self.dest_path + raw_info[i] + '_flowInfo.txt', res)
                        
                
            
            else: # Save info from MAT files
                
                mat_files = sorted(os.listdir(self.path))
                
                for mat_file in mat_files:
    
                    if mat_file[-3:] == 'mat': # Access only MAT files
                        
                        res = self.flowFromMAT(mat_file)
                        
                        if self.study == 'CKD2':

                            ind = mat_file.index('-')
                            
                            #ind_aux = mat_file.index('^')
                            
                            times = 2
                            
                            if '007' in mat_file or '011' in mat_file:
                                
                                times = 1
                            
                            for time in range(times):
                                
                                rep = '_' + str(time)

                                filename_save = self.dest_path + self.study + '_' + mat_file[:ind] + mat_file[24:26] + rep + '.flowInfo.txt'

                                np.savetxt(filename_save, res)
                            
                        #elif self.study == 'Hero' or self.study == 'Extr': # TO BE CHANGED

                            filename_save = self.path + mat_file[:-4].replace('msk_','') + '_flowInfo.txt'

                            #np.savetxt(filename_save, res)
                        
                        
        
        elif (self.flag == 'load') or (self.flag == 'Load') or (self.flag == 'LOAD'):
            
            # Load flow info from TXT files that have been previously saved

            if self.energy:
                
                mean_v, std_v, max_v, min_v, energy, area, net_flow, pos_flow, neg_flow = self.flowFromTXT() # Just an example!!
                
                return mean_v, std_v, max_v, min_v, energy, area, net_flow, pos_flow, neg_flow
                
            else:
                
                mean_v, std_v, max_v, min_v, area, net_flow, pos_flow, neg_flow = self.flowFromTXT() # Just an example!!
                
                return mean_v, std_v, max_v, min_v, area, net_flow, pos_flow, neg_flow
            
            # DO FURTHER ACTION IN HERE
 


                       
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