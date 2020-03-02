#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:47:41 2020

@author: andres
"""

import SimpleITK as sitk

import numpy as np

import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import os


class biasFieldCorrector:
    
    """
    Correct bias field inhomogeneities from a set of images in a given folder 
    with a given maximum number of iterations.
    
    Params:
        
        - path: main path where to find the images to correct (VTK files)
        - save_path: main path where to save the corrected images as VTK files
        - num_iter: maximum number of iterations for bias field correction (list with one integer)
    
    """
    
    def __init__(self, path, save_path, num_iter):
        
        self.path = path
        
        self.save_path = save_path
        
        self.num_iter = num_iter
    
    

    def biasFieldCorrection(self, img_array):
        
        """
        Correct bias field inhomogeneities from given array with a given maximum 
        number of iterations.
        
        Params:
            
            - inherited parameters from class (see description at the beginning of the class)
            - imag_array: Numpy array containing image to be corrected
            - num_iter: maximum number of iterations to apply
        
        Returns:
            
            - result: array with corrected bias
        
        """
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter() # Define corrector
            
        numberFittingLevels = 4
        
        corrector.SetMaximumNumberOfIterations( (self.num_iter)*numberFittingLevels  ) # Define maximum number of iterations
        
        if len(img_array.shape) == 2:
        
            img = sitk.GetImageFromArray(img_array) # Transform array into ITK image
            
            img = sitk.Cast(img, sitk.sitkFloat32) # Transform image values tp Float32
            
            output = corrector.Execute(img) # Execute corrector
            
            result = sitk.GetArrayFromImage(output) # Transform corrected image back into array
            
        elif len(img_array.shape) == 3:
            
            result = np.zeros(img_array.shape)
            
            for k in range(img_array.shape[-1]):
                
                img = sitk.GetImageFromArray(img_array[:,:,k]) # Transform array into ITK image
            
                img = sitk.Cast(img, sitk.sitkFloat32) # Transform image values tp Float32
                
                output = corrector.Execute(img) # Execute corrector
                
                result[:,:,k] = sitk.GetArrayFromImage(output) # Transform corrected image back into array
        
        return result



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

    def arrayNormalizer(self,array):
        
        """
        Normalizes array given between -1 and +1
        
        Removes potential outliers located in percentiles 1 or 99
        
        Params:
    
            - array: array to normalize
            
        Returns:
            
            - norm_array: normalized array without outliers
        
        """
        
        #perc99 = np.percentile(array, 99)
                    
        #perc1 = np.percentile(array, 1)
                    
        # Normalization
        
        norm_array = (2*(array- np.amin(array))/ (np.amax(array) - np.amin(array))) - 1
    
        return norm_array


    def array2vtk(self, array, filename, path, origin = [0,0,0], spacing = [1,1,1]):
                
        """
        Convert array into .vtk file
        
        - Params:
            inherited class parameters (see description at beginning of the class)
            array: array to be converted into .vtk file
            filename: filename with which to save array as VTK file
            path: path where to save VTK file
            origin: origin of coordinate system, by default (0,0,0)
            spacing: spacing of coordinate system, by default (1,1,1)
        
        """
          
        vtk_writer = vtk.vtkStructuredPointsWriter()
    
            
        # Check if destination folder exists
        
        #print('Checking if destination folder exists\n')
            
        isdir = os.path.isdir(path)
            
        if not isdir:
            
            os.makedirs(path)
            
            print('Non-existing destination path. Created\n')
        
        # Check if files already exist in destination folder
            
        exist = filename in os.listdir(path)
        
        overwrite = 'y'
        
        if exist:
            
            overwrite = input("File is already in folder. Do you want to overwrite? [y/n]\n")
        
        if overwrite == 'y' or overwrite == 'Y':
                
            vtk_writer.SetFileName(path + filename)
                
            vtk_im = vtk.vtkStructuredPoints()
        
            vtk_im.SetDimensions((array.shape[1],array.shape[0],array.shape[2]))
            
            vtk_im.SetOrigin(origin)
            
            vtk_im.SetSpacing(spacing)
        
            pdata = vtk_im.GetPointData()
        
            vtk_array = numpy_to_vtk(array.swapaxes(0,1).ravel(order='F'),deep=1)
        
            pdata.SetScalars(vtk_array)
        
            vtk_writer.SetFileType(vtk.VTK_BINARY)
        
            vtk_writer.SetInputData(vtk_im)
        
            vtk_writer.Update()
            
            #print('VTK file saved successfully!\n')
        
        else:
            print('\nOperation aborted\n')
            
            
    def __main__(self):        

        patients = os.listdir(self.path)
        
        # Access different patients
        
        for patient in patients:
            
            vtk_path = self.path + patient + '/'
        
            vtk_files = sorted(os.listdir(vtk_path))
            
            ind = [i for i, s in enumerate(vtk_files) if 'magBF' in s] # File indexes in mask folder
        
            for i in ind:
                
                if vtk_files[i][-3:] == 'vtk':
            
                    array, origin, spacing = self.readVTK(vtk_path, vtk_files[i])
                    
                    filename = vtk_files[i][:vtk_files[i].index('F') + 1] + '2' + vtk_files[i][vtk_files[i].index('F') + 1:]
                    
                    print(filename)
                    
                    array = self.arrayNormalizer(array)
                    
                    res = self.biasFieldCorrection(array)
                    
                    
                    #dest_path = self.save_path + study + '/'
                    
                    self.array2vtk(res, filename, vtk_path, origin, spacing)
                
                    
        
main_path = '/home/andres/Documents/_Data/_Patients/_Raw/_hero/'

save_path = '/home/andres/Documents/_Data/Bias_field_corrected_images/'

biasFieldCorr = biasFieldCorrector(main_path, save_path, [70])

biasFieldCorr.__main__()        

#dcm = pydicom.read_file('sQFLOW_BHdx')
#
#arr = dcm.pixel_array
#
#arr = arr/np.percentile(arr.flatten(),95)
#
#num_iter = [75]
#
#t1 = time.time()
#
#out = biasFieldCorrection(arr, num_iter)
#
#t2 = time.time()
#
#print('Elapsed time: {}'.format(t2-t1))
#
#plt.figure(figsize = (13,5))
#
#plt.subplot(1,3,1)
#
#plt.imshow(arr[0,:,:], cmap = 'gray')
#
#plt.colorbar()
#
#plt.subplot(1,3,2)
#
#plt.imshow(out[0,:,:], cmap = 'gray')
#
#plt.colorbar()
#
#plt.subplot(1,3,3)
#
#plt.imshow(abs(arr[0,:,:]-out[0,:,:]), cmap ='gray')
#
#plt.colorbar()