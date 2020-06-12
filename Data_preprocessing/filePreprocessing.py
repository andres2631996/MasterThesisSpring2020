#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:10:06 2020

@author: andres
"""

import numpy as np

import os

import sys

import vtk

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import time


class filePreprocessing:
    
    """
    Preprocessing raw VTK files. Set them to a common square matrix size and in
    case of magnitude images, normalize them between -1 and +1. Save them in 
    specified destination folder
    
    Params:
        
        - raw_path: raw folder where image files are organized into patients and
                    studies (str)
                    
        - dest_path: destination folder to save preprocessed files, being
                     organized into patients and studies (str)
                    
        - matrixSize: desired matrix size (list of int)
    
    Outputs:
        
        - Saved pre-processed files in destination folder
    
    """


    def __init__(self, raw_path, dest_path, matrixSize):
        
        self.raw_path = raw_path
        
        self.dest_path = dest_path
        
        self.matrixSize = matrixSize
    
    

    def existenceChecker(self, path, flag = 'folder'):
        
        """
        Checks that folder or file introduced exists.
        
        Params:
            
            - inherited from class (see class description above)
            
            - path: path to check if exists (str)
            
            - flag: if 'folder' checks for existence of a folder // if 'file' checks for existence of a file (str)
        
        Returns:
            
            - exists: flag telling if file or folder exists or not (bool)
        
        """
        
        if (flag == 'folder') or (flag == 'Folder') or (flag == 'FOLDER'):
        
            exists = os.path.isdir(path)
        
        elif (flag == 'file') or (flag == 'File') or (flag == 'FILE'):
            
            exists = os.path.isfile(path)
        
        else:
            
            print('Wrong flag introduced. Please introduce "folder" to check for folder existence or "file" to check for file existence')
    
            exists = None
        
        return exists



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


    def array2vtk(self, array, filename, dest_path, origin = [0,0,0], spacing = [1,1,1]):
                
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





    def zero_padding(self, mask, shape):
            
        """
        
        Zero pad mask image with shape of raw file. Can be used instead of np.pad
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            
            - mask: 2D array to zero-pad (array)

        
        Return: zero-padded mask in shape of raw file (result)
        
        """
        
        result = np.zeros(shape)
    
        center = np.array(np.array(shape)//2).astype(int) # Central voxel of raw file
        
        mask_half_shape = np.floor(np.array(mask.shape)/2).astype(int) # Half-shape of mask image
        
        # Take into account odd or even dimensions of array to zero pad
    
        if (np.remainder(mask.shape[0], 2) == 0) and (np.remainder(mask.shape[1], 2) == 0):
    
            result[center[0] - mask_half_shape[0] : center[0] + mask_half_shape[0], 
                   center[1] - mask_half_shape[1] : center[1] + mask_half_shape[1],:] = mask
    
        
        elif (np.remainder(mask.shape[0], 2) == 1) and (np.remainder(mask.shape[1], 2) == 0):
    
            result[center[0] - mask_half_shape[0] : center[0] + mask_half_shape[0] + 1, 
                   center[1] - mask_half_shape[1] : center[1] + mask_half_shape[1],:] = mask
                   
        elif (np.remainder(mask.shape[0], 2) == 0) and (np.remainder(mask.shape[1], 2) == 1):
    
            result[center[0] - mask_half_shape[0] : center[0] + mask_half_shape[0], 
                   center[1] - mask_half_shape[1] : center[1] + mask_half_shape[1] + 1,:] = mask
                   
        elif (np.remainder(mask.shape[0], 2) == 1) and (np.remainder(mask.shape[1], 2) == 1):
    
            result[center[0] - mask_half_shape[0] : center[0] + mask_half_shape[0] + 1, 
                   center[1] - mask_half_shape[1] : center[1] + mask_half_shape[1] + 1,:] = mask
               
        return result




    def arrayReshaping(self, array, bandRemoving):
        
        """
        Reshapes array to desired shape.
        
        Params:
    
            - array: array to be reshaped (must be 2D)
            
            - bandRemoving: two-element list indicating how many elements one wants 
                            to remove from left and right sides of the image. Usually
                            images have black bands left and right or fat deposits that
                            are useless and can be removed (list of 2 int)

        
        Returns:
            
            - reshaped: reshaped array with desired shape (2D) (array)
        
        """
        
        array = array[:,bandRemoving[0]: -bandRemoving[1],:] # Removal of left and right bands
        
        flag1 = flag2 = 0 # Parameters to control if array has odd dimensions
        
        # If dimensions of original array are odd, make them even for a simpler processing
        
        if np.remainder(self.matrixSize[0],2) == 1:
            
            # Odd desired shape
            
            flag1 = 1 # Remove later one row
            
            # Add 1 to make it even
            
            self.matrixSize[0] += 1
            
        
        if np.remainder(self.matrixSize[1],2) == 1:
            
            # Odd desired shape
            
            flag2 = 1 # Remove later one column
            
            # Add 1 to make it even
            
            self.matrixSize[1] += 1
        
        if np.remainder(array.shape[0], 2) == 1:
            
            # Odd shape of input array
            
            # Add extra row
            
            array = np.concatenate([array, np.zeros((1, array.shape[1], array.shape[2]))], axis = 0)
            
            
        if np.remainder(array.shape[1], 2) == 1:
            
            # Odd shape of input array
            
            # Add extra column
            
            array = np.concatenate([array, np.zeros((array.shape[0], 1, array.shape[2]))], axis = 1)   
            
            
        center = (np.array((array.shape))//2).astype(int) # Central coordinate of input array
        
        half_shape = (np.array((self.matrixSize))//2).astype(int) # Half-shape of input array
        
        if (array.shape[0] > self.matrixSize[0] or array.shape[0] == self.matrixSize[0]) and ((array.shape[1] > self.matrixSize[1] or array.shape[1] == self.matrixSize[1])):
            
            # Cropping both horizontally and vertically
            
            reshaped = array[center[0] - half_shape[0]: center[0] + half_shape[0], 
                             center[1] - half_shape[1]: center[1] + half_shape[1],:]
        
        elif (array.shape[0] < self.matrixSize[0]) and (array.shape[1] < self.matrixSize[1]):
            
            # Zero-padding both horizontally and vertically
            
            reshaped = self.zero_padding(array, [self.matrixSize[0], self.matrixSize[1], array.shape[2]] )
        
        elif (array.shape[0] < self.matrixSize[0]) and ((array.shape[1] > self.matrixSize[1] or array.shape[1] == self.matrixSize[1])):
            
            # Zero-padding horizontally and cropping vertically
            
            dif = np.array(self.matrixSize[0]) - np.array(array.shape[0]) # Number of rows to add
            
            half_dif = (np.floor(dif/2)).astype(int) # Number of rows to be added up and down
            
            if np.remainder(dif,2) == 1:
                
                # Add odd number of rows
            
                reshaped = np.concatenate([np.zeros((half_dif, array.shape[1], array.shape[2])),array, np.zeros((half_dif + 1, array.shape[1], array.shape[2]))], axis = 0)
                
            else:
            
                # Add even number of rows
                
                reshaped = np.concatenate([np.zeros((half_dif, array.shape[1], array.shape[2])),array, np.zeros((half_dif, array.shape[1], array.shape[2]))], axis = 0)
            
            # Crop vertically
            
            reshaped = reshaped[:, center[1] - half_shape[1]: center[1] + half_shape[1]]
        
        elif (array.shape[0] > self.matrixSize[0] or array.shape[0] == self.matrixSize[0]) and (array.shape[1] < self.matrixSize[1]):
            
            # Cropping horizontally and zero-padding vertically
            
            dif = np.array(self.matrixSize[1]) - np.array(array.shape[1]) # Number of columns to add
            
            half_dif = (np.floor(dif/2)).astype(int) # Number of columns to be added left and right
            
            if np.remainder(dif,2) == 1:
                
                # Add odd number of columns
            
                reshaped = np.concatenate([np.zeros((array.shape[0], half_dif, array.shape[2])),array, np.zeros((array.shape[0], half_dif + 1, array.shape[2]))], axis = 1)
                
            else:
            
                # Add even number of columns
                
                reshaped = np.concatenate([np.zeros((array.shape[0], half_dif, array.shape[2])),array, np.zeros((array.shape[0], half_dif, array.shape[2]))], axis = 1)
            
                # Crop horizontally
            
                reshaped = reshaped[center[0] - half_shape[0]: center[0] + half_shape[0],:,:]
        
        
        # Remove extra row or column added if 2D array had some odd dimension
        
        if flag1 == 1 and flag2 == 0: # Array has an odd number of rows, but it is now with an extra row
            
            reshaped = np.delete(reshaped, -1, axis = 0) # Delete extra row
        
        if flag1 == 0 and flag2 == 1: # Array has an odd number of columns, but it is now with an extra column
            
            reshaped = np.delete(reshaped, -1, axis = 1) # Delete extra column
                
        return reshaped       



    def arrayNormalizer(self,array):
        
        """
        Normalizes array given between -1 and +1
        
        Removes potential outliers located beyond percentile 99
        
        Params:
    
            - array: array to normalize
            
        Returns:
            
            - norm_array: normalized array without outliers
        
        """
        
        perc99 = np.percentile(array, 99)
                    
        #perc1 = np.percentile(array, 1)
                    
        # Normalization
        
        norm_array = (2*(array- np.amin(array))/ (perc99 - np.amin(array))) - 1
    
        return norm_array


    def __main__(self):
        
        #print('\nChecking if input folder exists\n') # Check if input folder exists

        exists_raw_folder = self.existenceChecker(self.raw_path, flag = 'folder')

        if exists_raw_folder:
            
            #print('Navigating through folders\n')
            
            studies = os.listdir(self.raw_path)
        
            for study in studies: # Go through all modalities
                
                study_path = self.raw_path + study +'/'
                
                patients = os.listdir(study_path)
                
                for patient in patients:
                    
                    patient_path = study_path + patient + '/'
                    
                    images = sorted(os.listdir(patient_path)) # Sorted in alphabetical order
        
                    for image in images:
                        
                        if image[-3:] == 'vtk': # Access only VTK files

                            if ('siemens' in study) and ('venc' not in image):
                        
                                band = [60, 60] # Remove empty bands in left and right sides of the images
                            
                            else:
                                
                                band = [30,30]
                            
                            array, origin, spacing = self.readVTK(patient_path, image) # Raw file reading
                            
                            # Array reshaping to desired dimensions
                            
                            #print('Array reshaping\n')
        
                            out = self.arrayReshaping(array, band)
                            
                            # Normalization only for magnitude images
                            
                            if 'mag' in image:
                            
                                # Normalization between -1 and +1
                                
                                #print('Array normalization\n')
                                
                                out = self.arrayNormalizer(out)
                            
                            # Decide name of final files
                            
                            if (matrixSize[0] < 128) or (matrixSize[0] == 128):
                            
                                key = '_crop' # Central FOV preprocessing pipeline
                            
                            else:
                                
                                key = '_prep' # Full FOV preprocessing pipeline
        
        
                            final_filename = image[:-4] + key + image[-4:]
                            
                            final_path = self.dest_path + study + '/' + patient + '/'
        
                            self.array2vtk(out, final_filename, final_path, origin, spacing) # Save preprocessed images as VTK files
                            
                            #print('Image pre-processed and saved successfully!\n')
                                
        else:
        
            print('\nNon-existing input folder. Please provide a valid input folder\n')                    


# Example of use            
            
#matrixSize = [64, 64] # Output matrix size
        
#raw_path = '/home/andres/Documents/_Data/_Patients/_Raw/' # Starting path

#dest_path = '/home/andres/Documents/_Data/_Patients/_Pre_crop/' # Destination path

#filePrep = filePreprocessing(raw_path, dest_path, matrixSize)

#t1 = time.time()

#filePrep.__main__()

#t2 = time.time()

#print('\nPre-processing time: {} seconds'.format(t2 - t1))





# To run from terminal:

# raw_path = sys.argv[1]
# dest_path = sys.argv[2] 
# matrixSize = sys.argv[3]
      
# if __name__ == "__main__":   









########### UNUSED FUNCTIONS. USEFUL FOR FUTURE #############


# Image Resizing function. UNUSED, but left if I need it in future occassions
    

#def imageResizing(filename, path, output_res, outputShape, bandRemoving):
#
#    """
#    Extracts image and spacing data in VTK file from a given folder and resizes
#    it so that it gets a specified resolution. In an intermediate step, the image
#    is cropped, removing empty left and right bands with zeros
#    
#    Params:
#        
#        - filename: image filename (must be VTK)
#        - path: folder where image is located
#        - output_res: output resolution (list of two elements)
#        - outputShape: desired shape to be provided (list of two elements)
#        - bandRemoving: two-element list indicating how many elements one wants 
#                        to remove from left and right sides of the image
#    
#    Outputs:
#        
#        - resized_array: resized image with output resolution
#
#    """        
#    
#    exists_folder = existenceChecker(path, 'folder') # Check that folder exists
#    
#    exists_file = existenceChecker(path + filename, 'file') # Check that filename exists
#    
#    if exists_folder and exists_file:
#        
#        if filename[-3:] == 'vtk': # Access only VTK files
#            
#            # Read VTK file
#            
#            array, origin, spacing = readVTK(path, filename)
#
#            new_array = array[:,bandRemoving[0]: -bandRemoving[1],:]
#
#            resize_factor = np.array([spacing[0], spacing[1]])/np.array(output_res)
#            
#            resized_array = np.zeros((int(np.round(new_array.shape[0]*resize_factor[0])), 
#                                                      int(np.round(new_array.shape[1]*resize_factor[1])),
#                                                      new_array.shape[2]))
#            
#            reshaped_array = np.zeros((outputShape[0], outputShape[1], array.shape[2]))
#            
#            #if resize_factor[0] < 2:
#
#            for k in range(array.shape[2]):
#            
#                resized_array[:,:,k] = skimage.transform.resize(new_array[:,:,k], 
#                                                         (resized_array.shape[0], resized_array.shape[1]), 
#                                                          order = 5)
#                
#                reshaped_array[:,:,k] = arrayReshaping(resized_array[:,:,k], outputShape)
#            
##            else:
##                
##                for k in range(array.shape[2]):
##                    
##                
##                    resized_array[:,:,k] = skimage.transform.rescale(new_array[:,:,k], 
##                                 (resize_factor[0], resize_factor[1]), order = 5, mode='reflect')
##                
##                    reshaped_array[:,:,k] = arrayReshaping(resized_array[:,:,k], outputShape)
#
#            return reshaped_array
#            
#            
#        else:
#            
#            print('File name introduced is not VTK')
#            
#    else:
#        
#        if not exists_folder and exists_file:
#            
#            print('\nFolder introduced is non-existent. Please introduce a valid folder\n')
#        
#        elif exists_folder and not exists_file:
#            
#            print('\nFile name introduced is non-existent. Please introduce a valid file name\n')
#        
#        else:
#            
#            print('\nFolder and file names introduced are non-existent. Please introduce a valid folder name and a valid file name\n')

# Histogram matching function. UNUSED, but left if I need it in future occassions
    

#def histogramMatching(image, reference):
#
#    """
#    Performs histogram matching of image w.r.t reference
#    
#    Params:
#        
#        - image: image whose histogram is modified
#
#        - reference: list with reference image array
#        
#    Returns:
#        
#        - result: image with matched histogram
#     
#    """
#    
#    # Verify that images have the same number of frames. 
#    
#    # Otherwise, do not perform any histogram matching
#    
#    if image.shape[2] == reference.shape[2]:
#        
#        result = np.zeros(image.shape)
#        
#        for k in range(image.shape[2]):
#            
#            result[:,:,k] = skimage.transform.match_histograms(image[:,:,k], reference[:,:,k])
#        
#        return result
#    
#    else:
#        
#        return image


# Cumulative histogram function. UNUSED, but left if I need it in future occassions

#def cumulativeHistogram(image, nbins = 40):
#    
#    """
#    Obtain cumulative histogram from image with a certain number of bins.
#    
#    Params:
#        
#        - image: image from where cumulative histogram is computed
#        
#        - nbins: number of bins for cumulative histogram (default: 40 bins)
#    
#    Returns:
#        
#        - res: cumulative histogram of image with specified number of bins
#    
#    
#    """
#
#    res = scipy.stats.cumfreq(image.flatten(), numbins = nbins, defaultreallimits=(-1, 1)) 
#    
#    n_voxels = np.prod(np.array(image.shape))
#    
#    res = res[0]/n_voxels
#    
#    return res

# Percentile Thresholding function. UNUSED, but left if I need it in future occassions 
    
#def percentileThresholding(image, thresh):
#    
#    """
#    Perform automatic thresholding on cropped image based on percentiles
#    
#    Params:
#        
#        - image: image to threshold
#
#        - thresh: percentile threshold (integer from 1 to 100)
#    
#    Returns:
#        
#        - res: thresholded image
#        
#        - mask: thresholding mask for corresponding phase image
#    
#    """
#    mask = np.zeros(image.shape)
#    
#    threshold = np.percentile(image.flatten(),thresh)
#    
#    positive_zone = np.where(image > threshold)
#
#    mask[positive_zone] = 1
#    
#    res = mask*image # Thresholding
#    
#    return res, mask            