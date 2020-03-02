#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:12:40 2020

@author: andres
"""

import numpy as np
import os 
import sys




class contourFileReader:
    
    """
    Read contour information from 2D or 3D images in TXT files and locate it in arrays with spatial and pixel coordinates.
    
    Params:
        - path: folder where .txt file is located
        - file: file with contour data
        - info_file: file with information on origin and spacing
    
    """
    
    def __init__(self, path, file, info_file):
        
        self.path = path
        self.file = file
        self.info_file = info_file
    


    def contourExtractor(self,origin, array, array_px):
        
        """
        Extract contour given its origin, the array with spatial coordinates and the array with pixel coordinates
        
        Params:
            - origin: origin
            - array: array with spatial coordinates
            - array_px: array with pixel coordinates
        
        """
        
        contour_list = [] # Final list where results will be saved, in spatial coordinates
        contour_list_px = [] # Final list where results will be saved, in pixel coordinates
        
        if origin.shape[0] > 2: # See if array is 2D or 3D
                
            # 3D array. Contours from all frames separated by [-1,-1]
            
            delimiter_index = np.array(np.where((array == (-1, -1)).all(axis=1)))[0]
            
            # Add 0 and -1 at beginning and end as delimiters
            
            delimiter_index = np.append(0, delimiter_index)
            
            delimiter_index = np.append(delimiter_index,-1)
            
            for i in range(delimiter_index.shape[0] - 1):
                
                if i == 0:
                    
                    aux = array[delimiter_index[i] : delimiter_index[i + 1],:]
                    aux_px = array_px[delimiter_index[i] : delimiter_index[i + 1],:]
                    
                    contour_list.append(aux)
                    contour_list_px.append(aux_px)
                    
                
                elif i > 0 and i < delimiter_index.shape[0] - 2: # All frames but last
            
                    contour_list.append(array[delimiter_index[i] + 1 : delimiter_index[i + 1],:])
                    contour_list_px.append(array_px[delimiter_index[i] + 1 : delimiter_index[i + 1],:])
                
                elif i == delimiter_index.shape[0] - 2: # Last frame
                    
                    aux = array[delimiter_index[i] + 1 : delimiter_index[i + 1],:]
                    aux_px = array_px[delimiter_index[i] + 1 : delimiter_index[i + 1],:]
                    
                    final = np.concatenate((aux, np.zeros((1,2)) + array[-1,:]))
                    final_px = np.concatenate((aux_px, np.zeros((1,2)) + array_px[-1,:]))
                    
                    contour_list.append(final)
                    contour_list_px.append(final_px)
        
        else:
            
            contour_list.append(array)
            contour_list_px.append(array_px)
            
        return contour_list, contour_list_px
    
    def spatial2pix(self, origin, spacing, array):
        
        """
        Convert array of spatial coordinates into array of pixel coordinates.
        
        Params:
            - origin: origin
            - spacing: spacing
            - array: array from where to extract the data
        
        """
        # Array transformation to pixel coordinates
                
        array_px = np.copy(array) # Array with pixel coordinates
        
        array_px = array - np.zeros((1,2)) - np.array([origin[0],origin[1]])
        
        array_px[:,0] = array[:,0]/spacing[0]
        array_px[:,1] = array[:,1]/spacing[1]
        
        return array_px
    
    def __main__(self):
        
        print('\nChecking if folder exists\n')
        isdir = os.path.isdir(self.path)
            
        if isdir:
        
            print('Checking if files exist\n')
            
            exist_data = self.file in os.listdir(self.path)
            
            exist_info = self.info_file in os.listdir(self.path)
            
            if exist_data and exist_info:
                
                print('Reading TXT files\n')
            
                whole_array = np.loadtxt(self.path + self.file)
                
                info_array = np.loadtxt(self.path + self.info_file)
                
                # Get origin and spacing from information file
                
                origin = info_array[0 : (info_array.shape[0]//2)]
                
                spacing = info_array[(info_array.shape[0]//2) : -1]
                
                spacing = np.append(spacing,info_array[-1])
                
                whole_array_px = self.spatial2pix(origin,spacing,whole_array)
                
                contour_list, contour_list_px = self.contourExtractor(origin, whole_array, whole_array_px)
                
                return contour_list, contour_list_px                
            
            elif exist_data and not exist_info:
                
                proceed = input('Information TXT files not available. Origin can be taken as (0,0,0) and spacing as (1,1,1). Would you like to proceed? [y/n]:')
                
                if proceed == 'y' or proceed == 'Y':
                    
                    whole_array = np.loadtxt(self.path + self.file)
                    
                    origin = np.array([0,0,0])
                    
                    spacing = np.array([1,1,1])
                    
                    contour_list, contour_list_px = self.contourExtractor(origin, whole_array, whole_array)
                    
                    return contour_list, contour_list_px
                
                else:
                    
                    print('Operation aborted by user as information files were lacking\n')
            
            else:
                print('No data files available. Perhaps you should look in another folder. Operation aborted\n') 
            
            
        
        else:
            
            print('Introduced path does not exist. Please introduce an existing folder\nOperation aborted\n')
            

#file_path = sys.argv[1] 
#file_name = sys.argv[2]
#info_file_name = sys.argv[3]
     
# if __name__ == "__main__"

orig_path = '/home/andres/Documents/_Data/Extra/_Contours/' # Folder for ground truths
#dest_path = '/home/andres/Documents/_Data/CKD_Part2/2_QFlow_Measurements/_Binary_masks/' # Folder for ground truths

files = sorted(os.listdir(orig_path))

file = files[1]
info_file = files[0]

cont2array = contourFileReader(orig_path, file, info_file)
a, b = cont2array.__main__()