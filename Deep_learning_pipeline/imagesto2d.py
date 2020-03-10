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



init_folder = '/home/andres/Documents/_Data/_Patients/'

final_folder = '/home/andres/Documents/_Data/_Patients2D/'

modalities = os.listdir(init_folder)

for modality in modalities:
    
    modality_path = init_folder + modality + '/'
    
    studies = os.listdir(modality_path)
    
    for study in studies:
        
        study_path = modality_path + study + '/'
        
        patients = os.listdir(study_path)
        
        for patient in patients:
            
            patient_path = study_path + patient + '/'
            
            images = os.listdir(patient_path)
            
            for image in images:
                
                image_array, origin, spacing = readVTK(image, patient_path)
                
                if 'magBF' in image:
                    
                        dest_path = patient_path.replace('_Patients','_Patients2D') + 'magBF/'
 
                elif 'mag_' in image:
                
                    dest_path = patient_path.replace('_Patients','_Patients2D') + 'mag/'
                
                
                elif 'pha' in image:
                
                    dest_path = patient_path.replace('_Patients','_Patients2D') + 'pha/'
                    
                
                elif 'msk' in image:
                
                    dest_path = patient_path.replace('_Patients','_Patients2D') + 'msk/'
                
                for i in range(image_array.shape[2]):
                    
                    filename = image[:-4] + '_frame' + str(i) + '.vtk'
                        
                    twoDarray2vtk(image_array[:,:,i], filename, dest_path, origin, spacing)
                
                if not('msk' in image):
                    
                    sum_filename = image[:-4] + '_sum.vtk'
                
                    twoDarray2vtk(np.sum(image_array, axis =2)/image_array.shape[2], sum_filename, dest_path, origin, spacing)