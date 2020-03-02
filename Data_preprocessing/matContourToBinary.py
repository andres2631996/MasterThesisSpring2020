#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:19:12 2020

@author: andres
"""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import vtk
import os
import sys
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import matplotlib.path as mplpath
import matplotlib.pyplot as plt


class contour2mask:
    
    """
    Convert ground-truth images from .mat files with known contours into masks.
    Images have to be 3D.
    
    If while segmenting, eddy currents have been corrected in the magnitude image,
    it also saves these corrected images.
    
    Params:
        - study: study to which images belong to
        - path: folder directory where mat file is located
        - filename: name of file
        - contour: if True provide image with gray contour, without filling
        - dest_path: folder where to leave the .vtk files
        - eddy_path: folder where to save magnitude images corrected in eddy currents
        
    
    """
    
    def __init__(self, study, path, filename, contour, dest_path, eddy_path):
        
        
        self.study = study
        self.path = path
        self.filename = filename
        self.contour = False
        self.dest_path = dest_path
        self.eddy_path = eddy_path

    

    def mat_loader(self):
        
        """
        Load .mat file from Segment software
        
        Params:
            - path and filename, inherited from the class
        
        Returns:
            - mat: dictionary with .mat file fields
            - st: structure field with data
            
        """
        
        #print('Conversion from .mat into array\n')
        
        mat = scipy.io.loadmat(self.path + self.filename) 
    
        st = mat.get('setstruct') 
        
        return mat, st
    
    
    def contourImage(self, gt_mag, roi_row, roi_col, central, central_int):
        
        """
        Return image with the contour without filling. Central pixel in the mask is highlighted
        
        Params:
            - path, filename and contour inherited from class (see description at the begining)
            - gt_mag: magnitude image
            - roi_row: contour row coordinates
            - roi_col: contour column coordinates
            - central: contour central point along all time frames
            - central_int: contour central coordinate (integer) along all time frames
            - bound: list with the minimum and maximum values of the contour, in pixel coordinates
    
        """
        #print('Computation of contour image\n')
        
        mask = np.zeros(gt_mag.shape)
        
        for k in range(roi_col.shape[-1]): # Loop through frames
            
            # Get absolute indexes for whole boundaries bound = (min_row, max_row, min_col, max_col)
            
            bound = [np.floor(np.amin(roi_row[:,k].flatten())),np.floor(np.amax(roi_row[:,k].flatten())),np.floor(np.amin(roi_col[:,k].flatten())),np.floor(np.amax(roi_col[:,k].flatten()))]
            
            bound = np.array(bound)
            
            #if k == 0:
                
                #print(bound)
    
            for i in np.linspace(bound[0],bound[1],bound[1]-bound[0]+1).astype(int): # Loop through rows
                
                for j in np.linspace(bound[2],bound[3],bound[3]-bound[2]+1).astype(int): # Loop through columns
                
                    cond1 = np.array(np.where(roi_row[:,k] > i))
                    cond2 = np.array(np.where(roi_row[:,k] < i + 1))
                    cond3 = np.array(np.where(roi_col[:,k] > j))
                    cond4 = np.array(np.where(roi_col[:,k] < j + 1))
                    
                    if (cond1.shape[0] != 0) and (cond2.shape[0] != 0) and (cond3.shape[0] != 0) and (cond4.shape[0] != 0):
                        
                        # Look for non-empty grids
                        
                        intersect_rows = np.intersect1d(cond1, cond2)
                        intersect_cols = np.intersect1d(cond3, cond4)
                        intersect_grid = np.intersect1d(intersect_rows, intersect_cols) # Indexes with points in same grid
                        
                        # Intersect_grid contains all the original contour points in the grid
                        
                        # Get the average point from all the presented points and the mean standard deviation
                        
                        point_avg = [np.mean(roi_row[intersect_grid, k]), np.mean(roi_col[intersect_grid, k])] # Average point in grid
                        
                        normal = central[k,:] - point_avg # Normal vector to contour
                        
                        if (normal[0] > 0 or normal[0] == 0) and (normal[1] > 0 or normal[1] == 0): # Look for right down area
                            
                            mask[j, i, k] = abs((i + 1 - point_avg[0]) * (j + 1 - point_avg[1]))
                        
                        elif (normal[0] > 0 or normal[0] == 0) and normal[1] < 0: # Look for left down area
                            
                            mask[j, i, k] = abs((i + 1 - point_avg[0]) * (point_avg[1] - j))
                        
                        elif normal[0] < 0 and (normal[1] > 0 or normal[1] == 0): # Look for right up area   
                        
                            mask[j, i, k] = abs((point_avg[0] - i) * (j + 1 - point_avg[1]))
                        
                        elif normal[0] < 0 and (normal[1] < 0): # Look for left up area
                            
                            mask[j, i, k] = abs((point_avg[0] - i) * (point_avg[1] - j))
            
            
        return mask
    
    
    def zero_padding(self, mask, shape):
            
        """
        
        Zero pad mask image with shape of raw file.
        
        Params:
            
            - inherited from class (check at the beginning of the class)
            
            - mask: 2D array to zero-pad

        
        Return: zero-padded mask in shape of raw file (result)
        
        """
        
        result = np.zeros(shape)
    
        center = np.array(np.array(shape)//2).astype(int) # Central voxel of raw file
        
        mask_half_shape = np.floor(np.array(mask.shape)/2).astype(int)
    
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
    
    

    def filledImage(self, gt_mag, roi_row, roi_col):
        
        
        """
        Fill image with contour.
        
        Params:
            - mask: contour image
            - gt_mag: original image
            - central_int: coordinate of central pixel in closed contour
            - bound: contour domain (min row, max row, min column, max column)
        
        Return: filled_img
        
        
        """
        #print('Filling contour\n')
        

        mask_filled = np.zeros(gt_mag.shape)
        
        for k in range(gt_mag.shape[-1]):
            
            coord = [] # List with ROI vertices
            
            for i in range(roi_col.shape[0]):
                
                coord.append((roi_row[i,k],roi_col[i,k]))
            
            # Create a grid with image coordinate points
            
            x, y = np.meshgrid(np.arange(gt_mag.shape[1]), np.arange(gt_mag.shape[0])) # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T
            
            p = mplpath.Path(coord) # make a polygon
            
            grid = p.contains_points(points) # Fill contour
            
            mask_filled[:,:,k] = grid.reshape((gt_mag.shape[0], gt_mag.shape[1])) # now you have a mask with points inside a polygon
            
            
        return mask_filled



    def array2vtk(self, array, final_filename, path, origin = [0,0,0], spacing = [1,1,1]):
        
        """
        Convert array into .vtk file
        
        - Params:
            inherited class parameters (see description at beginning of the class)
            array: array to be converted into .vtk file
            flag: parameter specifying if what is saved is a contour image or a binary mask
                'contour': contour is saved
                'binary': binary mask is saved
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

        overwrite = 'y'

        exist = final_filename in os.listdir(path)
            
        if exist:
    
            overwrite = input("File is already in folder. Do you want to overwrite? [y/n]")
        
        if overwrite == 'y' or overwrite == 'Y':
        
            vtk_writer.SetFileName(path + final_filename)
        
        else:
            
            print('\nOperation aborted\n')
            
            
        vtk_im = vtk.vtkStructuredPoints()
    
        vtk_im.SetDimensions((array.shape[1],array.shape[0],array.shape[2]))
        
        vtk_im.SetOrigin(origin)
        
        vtk_im.SetSpacing(spacing)
    
        pdata = vtk_im.GetPointData()
    
        vtk_array = numpy_to_vtk(array.swapaxes(0,1).ravel(order='F'),deep=1)
    
        pdata.SetScalars(vtk_array)
    
        #vtk_writer.SetFileType(VTK_BINARY)
    
        vtk_writer.SetInputData(vtk_im)
    
        vtk_writer.Update()
        
        
                

    def __main__(self):
        
        """
        Main pipeline. Returns the binary masks saved in .vtk file
        
        """
        
        mat, struct = self.mat_loader()
        
        y_shape = struct[0,0][10][0,0] # Full image shape
        x_shape = struct[0,0][9][0,0]

        gt_mag = struct[0,0][0] # Ground truth magnitude image (2D + t)
        #gt_phase = struct[0,1][0] # Ground truth phase image (2D + t)
        
        vendor = struct[0,0][32][0] # Scanner vendor, useful to see if images have been corrected in eddy currents
        
        col_res = struct[0,0][13][0,0] # Resolution of columns (mm)
        row_res = struct[0,0][14][0,0] # Resolution of rows (mm)
        time_res = struct[0,0][15][0,0] # Resolution in time (s)
        
        spacing = [col_res, row_res, time_res] # To save info into VTK file

        origin = []
        
        for i in range(len(struct[0,0][33][0])):
        
            origin.append(struct[0,0][33][0][i]) # To save info into VTK file
        
        mask = np.zeros(gt_mag.shape)

        for i in range(len(struct[0,0][64][0])): 
        
            roi_col = struct[0,0][64][0,i][0] # Indexes of ROI columns (pixels)
            roi_row = struct[0,0][64][0,i][1]  # Indexes of ROI rows (pixels)
    
            
            # Get central pixel in contour
            central = np.zeros((roi_col.shape[-1],2))
            central[:,0] = np.min(roi_row,0) + (np.max(roi_row,0) - np.min(roi_row,0))/2
            central[:,1] = np.min(roi_col,0) + (np.max(roi_col,0) - np.min(roi_col,0))/2
            
            # Get coordinate of central pixel along frames
            central_int = np.floor(central).astype(int)
    
            # Final mask with contour
            
            mask_cont = self.contourImage(gt_mag, roi_row, roi_col, central, central_int)
            
            # Computation of filled mask
            
            mask_filled = self.filledImage(gt_mag, roi_row, roi_col)
            
            
            final_mask = mask_cont + mask_filled
            
            # Final refinement: looj for values > 1 and substract them 1. 
            # Avoid some errors during filling
            
            ind = np.where(final_mask > 1)
            final_mask[ind] = final_mask[ind] - 1
            
            mask += final_mask

        
        # See if mask has desired shape, otherwise zero-pad it
        
        if final_mask.shape[0] != x_shape or final_mask.shape[1] != y_shape:
            
            mask = self.zero_padding(mask, (x_shape, y_shape))
            
        # Build final filename
        
        if 'dx' in self.filename:
                
            orient = 'dx'
        
        elif 'si' in self.filename:
            
            orient = 'si'
            
        
        if '_0' in self.filename:
            
            rep = '_0'
        
        elif '_1' in self.filename:
            
            rep = '_1'
            
        if study[0:-1] == 'CKD':
        
            final_filename = self.study + '_' + self.filename[:6] + '_' + self.filename[7:12] + orient + '_msk' + rep + '.vtk'
        
        elif study[0:-1] == 'Her':
            
            if not('venc' in self.filename): 
            
                final_filename = self.filename[:-5] + 'msk' + self.filename[-6:-4] + '.vtk'
            
            else:
                
                final_filename = self.filename[:-13] + 'msk' + self.filename[-14:-3] + 'vtk'
        
        elif study[0:-1] == 'Ext':
            
            final_filename = self.filename[:-5] + 'msk' + self.filename[-6:-4] + '.vtk'
        

        # Saving as vtk files in specified location
        
        if self.contour:
            
            self.array2vtk(mask_cont, final_filename, self.dest_path, origin, spacing)
            self.array2vtk(mask, final_filename, self.dest_path, origin, spacing)
        
        else:
            
            self.array2vtk(mask, final_filename, self.dest_path, origin, spacing)
            
        
            if vendor != 'Philips':
                
                if self.study == 'Hero':
                    
                    if not('venc' in self.filename): 
                
                        eddy_filename = self.filename[:-5] + 'magBF' + self.filename[-6:-4] + '.vtk'
                
                    else:
                    
                        eddy_filename = self.filename[:-13] + 'magBF' + self.filename[-14:-3] + 'vtk'
                    
                elif self.study == 'Extr':
                    
                    eddy_filename = self.filename[:-5] + 'magBF' + self.filename[-6:-4] + '.vtk'
                    
                else:
    
                    eddy_filename = self.study + '_' + self.filename[:6] + '_' + self.filename[7:12] + orient + '_magBF' + rep + '.vtk'
                
                # Save images corrected for eddy currents as VTK
                
                self.array2vtk(gt_mag, eddy_filename, self.eddy_path, origin, spacing)
                
            return mask, gt_mag
            
            
        

#gt_file = '/CKD007_MRI3_-2020-01-17_dx_AR.mat' # This will later on vary

#gt_path = '/home/andres/Documents/_Data/CKD_Part2/2_QFlow_Measurements/_Test_measurements/' # Folder for ground truths


study = 'Hero'
            
#gt_path = '/home/andres/Documents/_Data/Heroic/_Test_measurements_Andres/'

dest_path = '/home/andres/Documents/_Data/Extra/_Binary_masks/' # Folder for ground truths

gt_path = '/home/andres/Documents/_Data/Extra/_Test_measurements_Andres/'

#eddy_path = None

eddy_path = '/home/andres/Documents/_Data/Extra/_Eddy_corrected/'

frame = -1

eddy = False

for file in sorted(os.listdir(gt_path)):
    
    if file[-4:-1] == '.ma':
        
        #for j in range(2):
        
        cont2mask = contour2mask(study, gt_path, file, False, dest_path, eddy_path)
    
        mask, mag = cont2mask.__main__()
            
#        mag_center = (np.array(mag.shape)//2).astype(int)
#     
#        plt.figure(figsize = (13,5))
#        
#        plt.subplot(131)
#        
#        plt.imshow(mask[mag_center[0] - 30:mag_center[0] + 30,mag_center[1] - 30:mag_center[1] + 30,frame], cmap = 'gray')
#        
#        plt.subplot(132)
#        
#        plt.imshow(mag[mag_center[0] - 30:mag_center[0] + 30,mag_center[1] - 30:mag_center[1] + 30,frame], cmap = 'gray')
#        
#        plt.subplot(133)
#        
#        plt.imshow(mag[mag_center[0] - 30:mag_center[0] + 30,mag_center[1] - 30:mag_center[1] + 30,frame]*mask[mag_center[0] - 30:mag_center[0] + 30,mag_center[1] - 30:mag_center[1] + 30,frame], cmap = 'gray')
        

# study = sys.argv[1]
#file_path = sys.argv[2] 
#file_name = sys.argv[3]
#contour = sys.argv[4]
#dest_path = sys.argv[5]       
#shape = sys.argv[6] 

# if __name__ == "__main__":

    
    
    
    
    
    

        

    

        




