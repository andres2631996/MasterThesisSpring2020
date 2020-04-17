#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:02:04 2020

@author: andres
"""

import numpy as np

from vtk import vtkStructuredPointsWriter, vtkStructuredPointsReader, vtkStructuredPoints

from vtk import vtkMetaImageReader, vtkImageAppend, VTK_BINARY

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import sys

from scipy.ndimage.morphology import binary_erosion

import matplotlib.pyplot as plt

from skimage import measure

import os


class mask2contour:
    
    """
    Converts binary or quasi-binary masks from VTK files into contours.
    Contours are saved as txt files for all frames/slices (if 3D)
    Spacing and origin information are saved in an extra .txt file
    
    Use Marching Squares algorithm from Scikit-Image
    
    - Class parameters:
        - array: binary mask from where to extract contour
        - path: path where VTK file is located
        - dest_path: path where we want to save the contour
    
    """
    
    def __init__(self, array, path, dest_path):
        
        self.array = array
        self.path = path
        self.dest_path = dest_path
    


    def readVTK(self, order='F'):
        
        """
        Utility function to read vtk volume. 
        
        Params:
            - filename: VTK file name
        
        Returns:
            - numpy array
            - data origin
            - data spacing
        
        
        """
    
        reader = vtkStructuredPointsReader()
    
        reader.SetFileName(self.path + self.filename)
    
        reader.Update()
    
        image = reader.GetOutput()
    
        numpy_array = vtk_to_numpy(image.GetPointData().GetScalars())
    
        numpy_array = numpy_array.reshape(image.GetDimensions(),order='F')
    
        numpy_array = numpy_array.swapaxes(0,1)
    
        origin = list(image.GetOrigin())
    
        spacing = list(image.GetSpacing())
    
        return numpy_array, origin, spacing
    
    
    def save2txt(self, array, origin, spacing, it):
        
        """
        Save array as .txt file
        
        Params:
            - The ones inherited from the class
            - Array to save
            - Origin to save in txt file
            - Spacing to save in txt file
        
        """
        
        # Check if file exists in the destination path
        
        #print('Checking if destination folder exists\n')
        
        isdir = os.path.isdir(self.dest_path)
        
        if not isdir:
            
            os.makedirs(self.dest_path)
            
            print('Non-existing destination path. Created')
        
        #print('Checking if .txt file exists\n')
        
        files_in_folder = os.listdir(self.dest_path)
        
        # Name formatting
        
        ind_ = self.filename.find('.')
        
        
        final_filename = self.filename[0 : ind_] + '_ROIcontour_' + str(it) + '.txt'
        
        overwrite = 'y'
        
        #if final_filename in files_in_folder and not('OriginSpacing' in final_filename):
            
            
            #overwrite = input('The file already exists. Do you want to overwrite? [y/n]:\n')
            
        
        if overwrite == 'Y' or overwrite == 'y':
        
            np.savetxt(self.dest_path + final_filename, array)
             
            info_array = np.concatenate((np.array(origin),np.array(spacing)))
             
            np.savetxt(self.dest_path + self.filename[0 : ind_] + '_OriginSpacing.txt', info_array)
             
             #print('Contour(s) and information files saved successfully\n')
        
        else:
            
            print('Contour saving aborted\n')
        
        
        
        
        
        
    
    def __main__(self):
        
        
        # Go through all frames
        
        # Extract contours per frame
        
    
        
        #print('Extracting contours\n')

        
        if len(self.array.shape) == 3:
            
            cont = measure.find_contours(self.array[:,:,0], level = 1 - np.finfo(float).eps)

            for i in range(len(cont)):
        
                for k in range(self.array.shape[-1]):
                    
                    cont = measure.find_contours(self.array[:,:,k], level = 1 - np.finfo(float).eps)[i]

                    # Conversion into spatial coordinates. Take into account origin and spacing from VTK file
                    
                    #cont = np.array([origin[0], origin[1]]) + np.array([spacing[0], spacing[1]])*cont
                    
                    if k != self.array.shape[-1] - 1: # Except in last frame, add [-1,-1] as delimiter
                    
                        cont = np.concatenate((cont,np.zeros((1,2))-1)) # Concatenate contour coordinates with [-1,-1] to set [-1,-1] as delimiter between frames
                    
                    
                    if k == 0:
                        
                        whole_array = np.copy(cont) # Array to be saved in txt file
                    
                    else:
                        
                        whole_array = np.concatenate((whole_array, cont))
                    
                
                #self.save2txt(whole_array, origin, spacing, i)
                    
                return whole_array    
                    
                    
        elif len(array.shape) == 2:
            
            cont = measure.find_contours(array, level = 1.0 - np.finfo(float).eps)
            
            for i in range(len(cont)):
                
                whole_array = np.copy(cont[i])
                
                #self.save2txt(whole_array, origin, spacing, i)
                
            # Conversion into spatial coordinates. Take into account origin and spacing from VTK file
            
            #cont = np.array([origin[0], origin[1]]) + np.array([spacing[0], spacing[1]])*cont
            
            return whole_array
        
        else:
            
            print('\nWrong array shape. Please input a 2D or a 3D array')

    
    


#gt_path = '/home/andres/Documents/_Data/Extra/_Binary_masks/'

#dest_path = '/home/andres/Documents/_Data/Extra/_Contours/'

#gt_file = 'CKD007_MRI3_-2020-01-17_sin_AR_binaryMask.vtk'

#file = sys.argv[1] 
#gt_path = sys.argv[2]
#dest_path = sys.argv[3]       


# if __name__ == "__main__":

#files = sorted(os.listdir(gt_path))

#for file in files:
    
 #   if file[-4:-1] == '.vt': # Access only VTK files in origin folder

  #      mask2cont = mask2contour(file,gt_path,dest_path)
        
   #     final_cont = mask2cont.__main__()





