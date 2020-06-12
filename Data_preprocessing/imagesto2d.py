#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:36:30 2020

@author: andres
"""

import os 

import numpy as np

import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import shutil

# Code with useful functions to read a VTK file with a 2D+time volume and save its frames as separate 2D VTK files

def readVTK(filename, path, order='F'):
            
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
    
    
    
    
def twoDarray2vtk(array, filename, dest_path, origin = [0,0,0], spacing = [1,1,1]):
            
    """
    Convert 2D array into .vtk file
    
    - Params:
        
        array: array to be converted into .vtk file (array)
        
        filename: filename with which to save array as VTK file (str)
        
        origin: origin of coordinate system, by default (0,0,0) (list of 3 floats)
        
        spacing: spacing of coordinate system, by default (1,1,1) (list of 3 floats)
    
    """
      
    vtk_writer = vtk.vtkStructuredPointsWriter()

        
    # Check if destination folder exists
    
    #print('Checking if destination folder exists\n')
        
    isdir = os.path.isdir(dest_path)
        
    if not isdir:
        
        os.makedirs(dest_path)
        
        #print('Non-existing destination path. Created\n')
    
    # Check if files already exist in destination folder
        
    exist = filename in os.listdir(dest_path)
    
    overwrite = 'y'
    
    if exist:
        
        overwrite = input("File is already in folder. Do you want to overwrite? [y/n]\n")
    
    if overwrite == 'y' or overwrite == 'Y':
            
        vtk_writer.SetFileName(dest_path + filename)
            
        vtk_im = vtk.vtkStructuredPoints()
    
        vtk_im.SetDimensions((array.shape[1],array.shape[0], 1))
        
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
        
        

# Example of use

#init_folder = '/home/andres/Documents/_Data/_Patients/_Pre_crop/'

#final_folder = '/home/andres/Documents/_Data/_Patients2D/Pre_crop/'

#save_folder = '/home/andres/Documents/_Data/pha_problems/'

#studies = os.listdir(init_folder)

    
#for study in studies:
    
    #if study != '_ckd1':

     #   study_path = init_path + study + '/'

     #   patients = os.listdir(study_path)

     #   for patient in patients:

     #       patient_path = study_path + patient + '/'

     #       images = os.listdir(patient_path)

     #       for image in images:
                
     #           if 'msk' in image:

     #               image_array, origin, spacing = readVTK(image, patient_path)

     #               dest_path = patient_path.replace('_Patients','_Patients2D') + 'msk/'

     #               for i in range(image_array.shape[2]):

     #                   filename = image[:-4] + '_frame' + str(i) + '.vtk'
                        
     #                   print(dest_path + filename)
                        
                        #shutil.move(dest_path + filename, save_folder + filename)

                        #twoDarray2vtk(image_array[:,:,i], filename, dest_path, origin, spacing)

