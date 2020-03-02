#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:46:29 2020

@author: andres
"""

import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt
import vtk
import sys
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

class ExtractRawFiles:
    
    """
    Extract raw files from raw folders for studies CKD Part 1, CKD Part 2, 
    HEROIC and extra data. 
    
    Save data into 3D VTK files, being properly ordered.
    
    Params:
        
        - study: study that wants to be saved
        - path: folder from where to start looking for the data of each study
        - dest_path: folder where to save the "cleaned" raw images
        - key: key or set of keys (in list if more than one) used to differentiate PC-MR files from non-PC-MR files
        - v_enc: flag stating if velocity encoding is available
        - vein: flag starting if information on vein velocity is available
    
    
    """
    
    def __init__(self, study, path, dest_path, key, v_enc, vein):
        
        self.study = study
        
        self.path = path
        
        self.dest_path = dest_path
        
        self.key = key

    

    def array2vtk(self, array, filename, final_path, origin = [0,0,0], spacing = [1,1,1]):
            
        """
        Convert array into .vtk file
        
        - Params:
            inherited class parameters (see description at beginning of the class)
            array: array to be converted into .vtk file
            filename: filename with which to save array as VTK file
            final_path: folder where to save VTK file
            origin: origin of coordinate system, by default (0,0,0)
            spacing: spacing of coordinate system, by default (1,1,1)
        
        """
          
        vtk_writer = vtk.vtkStructuredPointsWriter()
    
            
        # Check if destination folder exists
        
        #print('Checking if destination folder exists\n')
            
        isdir = os.path.isdir(final_path)
            
        if not isdir:
            
            os.makedirs(final_path)
            
            print('Non-existing destination path. Created\n')
        
        # Check if files already exist in destination folder
            
        exist = filename in os.listdir(final_path)
        
        overwrite = 'y'
        
        if exist:
            
            overwrite = input("File is already in folder. Do you want to overwrite? [y/n]\n")
        
        if overwrite == 'y' or overwrite == 'Y':
                
            vtk_writer.SetFileName(final_path + filename)
                
            vtk_im = vtk.vtkStructuredPoints()
        
            vtk_im.SetDimensions((array.shape[1],array.shape[0],array.shape[2]))
            
            vtk_im.SetOrigin(origin)
            
            vtk_im.SetSpacing(spacing)
        
            pdata = vtk_im.GetPointData()
        
            vtk_array = numpy_to_vtk(array.swapaxes(0,1).ravel(order='F'),deep=1, array_type=vtk.VTK_FLOAT)
        
            pdata.SetScalars(vtk_array)
        
            vtk_writer.SetFileType(vtk.VTK_BINARY)
        
            vtk_writer.SetInputData(vtk_im)
        
            vtk_writer.Update()
            
            #print('VTK file saved successfully!\n')
        
        else:
            print('\nOperation aborted\n')
            
    

    def orientationExtractor(self, folder):

        """
        Extract whether files to be read belong to right or left kidney.
        
        Params:
            - inherited from class (see description at beginning)
            - folder: folder name where to check for information
            
        Return: orientation ('dx' if right, 'si' if left)

        """ 
        
        if ('dx' in folder) or ('DX' in folder) or ('Right' in folder):
            
            orientation = 'dx'
        
        elif ('si' in folder) or ('SI' in folder) or ('Left' in folder):
            
            orientation = 'si'
        
        return orientation
    
    
    def repetitionExtractor(self, folder):
        
        """
        Extract whether files to be read are repeated acquisitions or not.
        
        Params:
            - inherited from class (see description at beginning)
            - folder: folder name where to check for information
            
        Return: repetition string (rep) ('_0' for first acquisition, '_1' for second... and so on)

        """ 
        
        for i in range(6):
            
            test_str = '_' + str(i)
            
            if test_str in folder:
            
                rep = '_' + str(i)
                
                break
            
            else:
                
                rep = '_0'
        
        return rep
        
    
    def modalityExtractor(self, string):
        
        """
        Extract image modality from string to read.
        
        Params:
            - inherited from class (see description at beginning)
            - string: string list where to check for information
            
        Return: string with image modality ('mag' for magnitude, 'pha' for phase, 'oth' for others)

        """
        modality = 'None'
        
        if (('M' in string) or ('FFE' in string)) and not ('PCA' in string):
            
            modality = 'mag'
        
        elif ('P' in string) or ('VELOCITY MAP' in string) or ('PHASE CONTRAST M' in string):
            
            modality = 'pha'
        
        elif (('PCA' in string) or ('MAG' in string)) and not('FFE' in string):
            
            modality = 'oth'
            
        return modality
    
    
    def vencExtractor(self, files_folder):
        
        """
        Extract velocity encoding information from string to read
        
        Params:
            - files_folder: string to read from where information is extracted
        
        Return: flag with velocity encoding value
        
        """
        
        flag_venc = False
        
        venc = '100'
        
        if '80' in files_folder:
                                    
            flag_venc = True
            
            venc = '80'
            
        elif '100' in files_folder:
            
            flag_venc = True
            
            venc = '100'
        
        elif '120' in files_folder:
            
            flag_venc = True
            
            venc = '120'
        
        return venc, flag_venc
    
    
    def arrayReformatting(self, array):
        
        """
        Helps to reformat array from Time-First dimension to Time-Last dimension.
        It also swaps rows with columns (90 degrees image flipping)
        
        Params:
            
            - inherited from class (see description at beginning)
            - array: array to reformat
        
        Returns:
            
            - reformat_array: reformatted array
            
        """
        reformat_array = np.swapaxes(array, 0, 2) # Swap axes 0 and 2 for time dimension
                                                    
        reformat_array = np.swapaxes(reformat_array, 0, 1) # Now swap axes 0 and 1 (image flipping) 
        
        return reformat_array
    
        
    
    def dicomFieldExtractor(self, image_path):
        
        """
        Extract DICOM field information from given DICOM file
        
        Params:
            - inherited from class (see description at beginning)
            - image_path: DICOM file path from where to extract info
        
        Returns:
            - If DICOM files are 3D --> returns 3D array (array), orientation indicator (orientation) 
              and repetition indicator (rep)
            
            - If DICOM files are 2D --> returns arrays with all pixel arrays (arrays_array), times (time_array),
              modalities (modal_array) and instance numbers found (instance_array)
        
        """
        
        #print('Extracting relevant information from found DICOM files\n')
        

        if not os.path.isdir(image_path):

            dcm = pydicom.read_file(image_path) # Dataset with all DICOM fields
            
            keys = list(dcm.keys()) # DICOM tags
            
            key_str = [] # DICOM tags as strings
            
            for i in range(len(keys)):
                
                key_str.append(str(keys[i]))

            array = dcm.pixel_array
            
            if ('(0028, 1053)' in key_str) and ('(0028, 1052)' in key_str): # See if rescale slope and rescale intercept are DICOM tags
            
                array = array*float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

            array = self.arrayReformatting(array)
            
            orientation = self.orientationExtractor(image_path)
            
            rep = self.repetitionExtractor(image_path)
            
            vein_flag = False
            
            if 'ein' in image_path:
                                    
                # Decide on vein acquisition
                
                vein_flag = True

            return array, orientation, rep, vein_flag
        
        else:
            
            image_files = os.listdir(image_path) # Set of image files
 
            times = [] # List with all acquisition times
                                        
            num = [] # List with acquisition numbers
            
            modal = [] # List with modalities
            
            arrays = [] # List with image arrays
            
            instance_num = [] # List with different instance numbers

            for image in image_files: # Go through all files
                
                dcm = pydicom.read_file(image_path + image) # Dataset with all DICOM fields
                
                keys = list(dcm.keys()) # DICOM tags
                
                key_str = [] # DICOM tags as strings
                
                for i in range(len(keys)):
                    
                    key_str.append(str(keys[i]))
                

                if ('(0028, 1053)' in key_str) and ('(0028, 1052)' in key_str): # See if rescale slope and rescale intercept are DICOM tags
  
                    arrays.append(dcm.pixel_array*float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)) # Array where to save pixel arrays
                
                else:
                    
                    arrays.append(dcm.pixel_array)
    
                num.append(float(dcm.AcquisitionNumber)) # Different acquisition numbers
        
                times.append(float(dcm.TriggerTime)) # Acquisition time

                modal.append(self.modalityExtractor(dcm.ImageType)) # List with all modalities

                instance_num.append(float(dcm.InstanceNumber))

            spacing = dcm.PixelSpacing # Resolution in mm/pixel (for VTK file)
                    
            thickness = float(dcm.SliceThickness) # Slice thickness in Z
            
            spacing.append(thickness) # Append this to spacing as third dimension (for VTK file)
            
            origin = dcm.ImagePositionPatient # Origin in mm (for VTK file)
            
            arrays_array = np.array(arrays) # 3D array with all pixel arrays found
            
            reformat_array = self.arrayReformatting(arrays_array) # Reformatted 3D array into desired 
            
            num_array = np.array(num) # Array with all acquisition numbers found
            
            time_array = np.array(times) # Array with all times found
            
            modal_array = np.array(modal) # Array with all modalities found
            
            instance_array = np.array(instance_num) # Array with all instance numbers found
            
            num_folder = float(dcm.AcquisitionNumber) # Acquisition number for the last image read in the folder. All images in the same folder have the same acquisition number
            
            return reformat_array, num_array, time_array, modal_array, instance_array, spacing, origin, num_folder

                                    
                                     
    def __main__(self):      

        if self.study == 'CKD1':
            
            #print('\nNavigating through folders\n')
            
            # Extract raw files from CKD Part 1 study
            
            #key = 'QFLOW' # Key to look for folders with images
            
            patients = os.listdir(self.path)
            
            for patient_folder in patients:
                
                patientID = patient_folder[0:11]
                
                # Dive into patient folders
                
                if patient_folder[0:3] == 'CKD': # Check that folder belongs to study
                    
                    patient_files = os.listdir(self.path + patient_folder + '/')
                    
                    for image_folder in patient_files:
                        
                        if self.key[0] in image_folder:
                        
                            image_path = self.path + patient_folder + '/' + image_folder + '/' # Image folder

                            orientation = self.orientationExtractor(image_folder)
                            
                            if patientID != 'CKD052_MRI1' or orientation != 'si': # Remove left acquisition of CKD052: PROBLEMATIC
                                
                                # Extract DICOM fields
                                
                                arrays_array, num_array, time_array, modal_array, _ , spacing, origin, _ = self.dicomFieldExtractor(image_path)
        
                                num_unique = np.unique(num_array) # Array with unique acquisition numbers

                                mod_unique = np.unique(modal_array) # Array just with the different modalities
        
                                cont_acq = 0 # Counter for repeted acquisitions
                                
                                #print('Saving as VTK files with relevant information extracted from DICOM files\n')
                                
                                for acq_num in num_unique:
                                    
                                    ind = np.where(num_array == acq_num) # Images with a certain acquisition number
                                    
                                    # Look for the different modalities under the same acquisition number
                                    
                                    for m in mod_unique:
                                        
                                        # Images from same modality and same acquisition number
                                        
                                        # Get index of these images in overall array
                                        
                                        ind_m = np.where(modal_array == m) # Indexes of images of same modality
                                        
                                        ind_num_m = np.intersect1d(np.array(ind),np.array(ind_m)) # Indexes of images of same modality and same acquisition number
                                        
                                        time_num_m = time_array[ind_num_m] # Time values for same modality and same acquisition number
                                        
                                        time_num_m_sort_ind = np.argsort(time_num_m) # Indexes of sorted time frames
                                        
                                        final_image = arrays_array[:,:,ind_num_m[time_num_m_sort_ind]] # Sorted final image

                                        # Structure of final filename: study + patient + right/left kidney + magnitude/phase/other + repeated acquisition
                                        
                                        final_filename = study + '_' + patientID + '_' + orientation + '_' + m + '_' + str(cont_acq) + '.vtk'
                                        
                                        # Save as VTK file in provided destination folder
                                        
                                        self.array2vtk(final_image, final_filename, self.dest_path + '_' + m + '/', origin, spacing)
                                             
                                    cont_acq += 1  
                                            
                                            
        elif self.study == 'CKD2':
            
            #print('\nNavigating through folders\n')
            
            patients = os.listdir(self.path) 
        
            for patient_folder in patients:
        
                patientID = patient_folder[0:11]                                   
                                            
                if patient_folder[0:3] == 'CKD': # Check that folder belongs to study
                        
                    patient_files = sorted(os.listdir(self.path + patient_folder + '/'))
                    
                    for image_folder in patient_files:
                        
                        if self.key[0] in image_folder:
                            
                            image_path = part2_path + patient_folder + '/' + image_folder + '/' # Image folder
                        
                            orientation = self.orientationExtractor(image_folder)
                            
                            rep = self.repetitionExtractor(image_folder)
                            
                            arrays_array, num_array, time_array, mod_array, instance_array, spacing, origin, num_folder = self.dicomFieldExtractor(image_path)
       
                            num_unique = np.unique(num_array) # Array with unique acquisition numbers
                               
                            mod_unique = np.unique(mod_array) # Array just with the different modalities
    
                            #cont = np.where(num_unique == num_folder) # Images with a certain acquisition number
                                    
                            # Look for the different modalities under the same acquisition number
                            
                            #print('Saving as VTK files with relevant information extracted from DICOM files\n')

                            for m in mod_unique:
                                
                                # Images from same modality and same acquisition number
                                
                                # Get index of these images in overall array
                                
                                ind_m = np.array(np.where(mod_array == m))[0] # Indexes of images of same modality

                                time_m = time_array[ind_m] # Time values for same modality and same acquisition number
                                
                                time_m_sort_ind = np.argsort(time_m) # Indexes of sorted time frames
                                
                                final_image = arrays_array[:,:,ind_m[time_m_sort_ind]] # Sorted final image
                                
                                # Structure of final filename: study + patient + right/left kidney + magnitude/phase/other + repeated acquisition
                                
                                final_filename = study + '_' + patientID + '_' + orientation + '_' + m + rep + '.vtk'
                                
                                # Save as VTK file in provided destination folder
                                
                                self.array2vtk(final_image, final_filename, self.dest_path + '_' + m + '/', origin, spacing)
         
                                                
                                                    
        elif (self.study == 'hero') or (self.study == 'Hero') or (self.study == 'HERO'):
            
            #print('\nNavigating through folders\n')
        
            # Files from HEROIC studies
        
            individuals = os.listdir(self.path)
        
            for individual in individuals: # Go through healthy volunteers and pilot patients
        
                individual_path = self.path + individual + '/' # Healthy or patient
        
                vendors = os.listdir(individual_path) # Vendor: GE or Siemens
        
                for vendor in vendors: # Go through the different vendors
                    
                    vendor_path = individual_path + vendor + '/' # Vendor folder
                    
                    patients = os.listdir(vendor_path) # List of patients
                    
                    for patient in patients: # Go through all patients
                        
                        patient_path = vendor_path + patient + '/' # Patient folder
                        
                        patient_files = os.listdir(patient_path) # Patient images
                        
                        if 'GE' in vendor: # Access GE files

                            for files_folder in patient_files: # Access folder with DCM images
                                    
                                files_path = patient_path + files_folder + '/'
                    
                                if (self.key[0] in files_folder) or (self.key[1] in files_folder): # Access only PC-MRI
                                    
                                    orientation = self.orientationExtractor(files_folder) # Orientation info
                                    
                                    rep = self.repetitionExtractor(files_folder) # Repetition info

                                    arrays_array, _ , time_array, _ , instance_array, spacing, origin, _ = self.dicomFieldExtractor(files_path)
    
                                    ind_mag = np.array(np.where(instance_array > 40))[0] # Indices with magnitude images
                                    
                                    ind_phase = np.array(np.where(instance_array < 41))[0] # Indices with phase images
                                    
                                    time_mag = time_array[ind_mag] # Times for magnitude images
                                    
                                    time_phase = time_array[ind_phase] # Times for phase images
                                    
                                    # Get indices of ordered times
                                    
                                    argsort_mag = np.argsort(time_mag) # For magnitude images
                                    
                                    argsort_phase = np.argsort(time_phase) # For phase images
                                    
                                    mag_image = arrays_array[:,:,ind_mag[argsort_mag]] # Final magnitude image
                                    
                                    phase_image = arrays_array[:,:,ind_phase[argsort_phase]] # Final phase image
                                    
                                    mag_filename = study + '_' + patient + '_' + orientation + '_mag' + rep + '.vtk'
                                    
                                    phase_filename = study + '_' + patient + '_' + orientation + '_pha' + rep + '.vtk'
                                    
                                    # Save as VTK file in provided destination folder
                                    
                                    #print('Saving as VTK files with relevant information extracted from DICOM files\n')
                                    
                                    # Save magnitude image
                                    
                                    self.array2vtk(mag_image, mag_filename, self.dest_path + '_GE/_mag/', origin, spacing)
                                    
                                    # Save phase image
                                    
                                    self.array2vtk(phase_image, phase_filename, self.dest_path + '_GE/_pha/', origin, spacing)
                                            
                                            
                        if 'Siemens' in vendor: # Access GE files

                            for files_folder in patient_files: # Access folder with DCM images
                                    
                                files_path = patient_path + files_folder + '/'
                                
                                if self.key[2] in files_folder: # Access only PC-MRI
                                    
                                    #venc_available = False # State if v_enc is available or not, so to add it to the final filename
                                    
                                    # Study cases with different encoding velocities
                                    
                                    venc, flag_venc = self.vencExtractor(files_folder)
                                    
                                    # Decide on orientation
                                    
                                    orientation = self.orientationExtractor(files_folder)
                                    
                                    # Decide on repetition
                                    
                                    rep = self.repetitionExtractor(files_folder)
                                    
                                    # Extract DICOM fields of all files as lists
                                    
                                    arrays_array, _, time_array, modal_array, _ , spacing, origin, _ = self.dicomFieldExtractor(files_path)
       
                                    mod_unique = np.unique(modal_array)

                                    argsort_time = np.argsort(time_array) # Array with sorted time indices
                                            
                                    final_image = arrays_array[:,:, argsort_time] # Final phase image
                                            
                                    if flag_venc:
                                    
                                        final_filename = study + '_' + patient + '_' + orientation + '_' + mod_unique[0] + rep + '_venc' + venc + '.vtk'
                                    
                                    else:
                                        
                                        final_filename = study + '_' + patient + '_' + orientation + '_' + mod_unique[0] + rep + '.vtk'
                                    
                                    
                                    #print('Saving as VTK files with relevant information extracted from DICOM files\n')
                                    
                                    self.array2vtk(final_image, final_filename, self.dest_path + '_Siemens/_' + mod_unique[0] + '/', origin, spacing)
        
        
                                elif self.key[3] in files_folder: # Access only PC-MRI
                                    
                                    image_folders = os.listdir(files_path)
                                
                                    for new_folder in image_folders:
                                        
                                        if self.key[3] in new_folder: # Access only Qflow images
                                            
                                            rep = self.repetitionExtractor(files_folder)
                                            
                                            new_path = files_path + new_folder + '/'
                                            
                                            reformat_array, num_array, time_array, modal_array, instance_array, spacing, origin, num_folder = self.dicomFieldExtractor(new_path)

                                            mod_unique = np.unique(modal_array)
                                             
                                            argsort_time = np.argsort(time_array) # Array with sorted time indices
                                            
                                            final_image = arrays_array[:,:, argsort_time] # Final phase image
                                            
                                            final_filename = study + '_' + patient + '_' + orientation + '_' + mod_unique[0] + rep + '.vtk'
                                            
                                            #print('Saving as VTK files with relevant information extracted from DICOM files\n')
                                            
                                            self.array2vtk(final_image, final_filename, self.dest_path + '_Siemens/_' + mod_unique[0] + '/', origin, spacing)
        
        
        elif (self.study == 'extr') or (self.study == 'Extr') or (self.study == 'EXTR'):
            
            #print('\nNavigating through folders\n')
            
            # Extra files
            
            patients = os.listdir(self.path) # List of extra patients
            
            for patient_folder in patients: # Go through patients
                
                if patient_folder[0:3] == 'CKD': # Verify the patient
                    
                    patientID = patient_folder[0:12]
                    
                    patient_path = self.path + patient_folder + '/'
        
                    image_folders = os.listdir(patient_path)

                    for image_folder in image_folders: 

                        # Enter only folders with images

                        if image_folder == 'DICOM':
                            
                            # Get images inside folder
                            
                            new_path = patient_path + image_folder + '/'
                            
                            images = os.listdir(new_path)
                            
                            for image in images:
                                
                                if self.key[0] in image:

                                    array, orientation, rep, vein_flag = self.dicomFieldExtractor(new_path + image)
        
                                    # Obtention of magnitude, phase and other modalities
                                    
                                    mag_img = array[:,:, 0 : array.shape[2]//3] # Magnitude image array
                                    
                                    oth_img = array[:,:, (array.shape[2]//3):(2*array.shape[2]//3)] # Extra image array
                                    
                                    pha_img = array[:,:, (2*array.shape[2]//3) : array.shape[2]] # Phase image array
                                    
                                    pha_img = 0.034*pha_img - 70 # Adjust by rescale slope and rescale intercept
                                    
                                    origin = [118.69, -77.924, 140.906]
                                    
                                    spacing = [1.0069, 1.0069, 8]
                                    
                                    if vein_flag:
                                        
                                        mag_filename = study + '_' + patientID + '_' + orientation + '_mag' + rep + '_vein.vtk'
                                        
                                        oth_filename = study + '_' + patientID + '_' + orientation + '_oth' + rep + '_vein.vtk'
                                        
                                        pha_filename = study + '_' + patientID + '_' + orientation + '_pha' + rep + '_vein.vtk'
                                        
                                    else:
                                    
                                        mag_filename = study + '_' + patientID + '_' + orientation + '_mag' + rep + '.vtk'
                                        
                                        oth_filename = study + '_' + patientID + '_' + orientation + '_oth' + rep + '.vtk'
                                        
                                        pha_filename = study + '_' + patientID + '_' + orientation + '_pha' + rep + '.vtk'
        

                                    #print('Saving as VTK files with relevant information extracted from DICOM files\n')
                                     
                                    self.array2vtk(mag_img, mag_filename, self.dest_path + '/_mag/', origin, spacing) # Save magnitude images
                                    
                                    self.array2vtk(oth_img, oth_filename, self.dest_path + '/_oth/', origin, spacing) # Save extra images
                                    
                                    self.array2vtk(pha_img, pha_filename, self.dest_path + '/_pha/', origin, spacing) # Save phase images
                            
                            
                        if self.key[1] in image_folder:
                            
                            new_path = patient_path + image_folder + '/'
                            
                            arrays_array, num_array, time_array, modal_array, instance_array, spacing, origin, num_folder = self.dicomFieldExtractor(new_path)
                            
                            ind_mag = np.array(np.where(instance_array > 30))[0] # Indices with magnitude images
                            
                            ind_phase = np.array(np.where(instance_array < 31))[0] # Indices with phase images
                            
                            time_mag = time_array[ind_mag] # Times for magnitude images
                            
                            time_phase = time_array[ind_phase] # Times for phase images
                            
                            # Get indices of ordered times
                            
                            argsort_mag = np.argsort(time_mag) # For magnitude images
                            
                            argsort_phase = np.argsort(time_phase) # For phase images
                            
                            mag_image = arrays_array[:,:, ind_mag[argsort_mag]] # Final magnitude image
                            
                            phase_image = arrays_array[:,:, ind_phase[argsort_phase]] # Final phase image
                            
                            # Decide on orientation
                                    
                            orientation = self.orientationExtractor(image_folder)
                            
                            # Decide on repetition
                            
                            rep = self.repetitionExtractor(image_folder)

                            mag_filename = study + '_' + patientID + '_' + orientation + '_mag' + rep + '.vtk'
                            
                            phase_filename = study + '_' + patientID + '_' + orientation + '_pha' + rep + '.vtk'


                            # Save as VTK file in provided destination folder
                            
                            #print('Saving as VTK files with relevant information extracted from DICOM files\n')
                            
                            # Save magnitude image
                            
                            self.array2vtk(mag_image, mag_filename, self.dest_path + '/_mag/', origin, spacing)
                            
                            # Save phase image
                            
                            self.array2vtk(phase_image, phase_filename, self.dest_path + '/_pha/', origin, spacing)
                                        


# Important folders

part1_path = '/home/andres/Documents/_Data/CKD_Part1/' # Folder with CKD Part 1 study info

part2_path = '/home/andres/Documents/_Data/CKD_Part2/1_Raw_data/' # Folder with CKD Part 2 study info

extra_path = '/home/andres/Documents/_Data/Extra/'

heroic_path = '/home/andres/Documents/_Data/Heroic/'

save_part1_path = '/home/andres/Documents/_Data/CKD_Part1/Raw_data/' # Folder where to save files on CKD Part 1

save_part2_path = '/home/andres/Documents/_Data/CKD_Part2/3_Clean_Data/' # Folder where to save files on CKD Part 2

save_heroic_path = '/home/andres/Documents/_Data/Heroic/Clean_data/' # Folder where to save files on Heroic

save_extra_path = '/home/andres/Documents/_Data/Extra/Clean_data/' # Folder where to save files on Extra


# Important input variables

#study = 'CKD1' # Name of study from where we want to recover the raw files

#key_study1 = ['QFLOW']

#study = 'CKD2'

#key_study2 = ['QFLOW']

#study = 'Hero'

#key_hero = ['PC', 'QFLOW', 'QFLOW', 'Qflow']

study = 'Extr'

key_extra = ['QFLOW', 'PC']   

extract = ExtractRawFiles(study, extra_path, save_extra_path, key_extra, True, True)
extract.__main__() 

# study = sys.argv[1]
#file_path = sys.argv[2] 
#dest_path = sys.argv[3]
# key = sys.argv[4]       


# if __name__ == "__main__":                             