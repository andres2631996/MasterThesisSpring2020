#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:35:02 2020

@author: andres
"""
import numpy as np

import os

import pydicom

import sys

from flowStatistics import figure_saving, linear_regression_test


class TimeAnalyzer:
    
    """
    Access data information on time (only Philips and GE files). Get information
    on time resolution, patient heart rate and expected heart rate computed from
    time resolution values.
    
    Params:
        
        - paths: list of data paths where to look for time information
        - studies: list of studies to apply
        - folder_keys: list of keys to use for different studies in folders
        - dest_path: folder where to save time information as TXT files
        - times: list with times
        - bpm: list with bpm values
        - bpm_comp: list with computed bpm values
        - file_info: list with file information
        - flag: decide to save or to load existing information
    
    Returns:
        
        - saved TXT files with time information in desired destination path
        - linear regression analysis of bpm vs computed bpm 

    
    """
    
    def __init__(self, paths, studies, folder_keys, dest_path, times, bpm, bpm_comp, file_info, flag):
        
        self.paths = paths
        
        self.studies = studies
        
        self.folder_keys = folder_keys
        
        self.dest_path = dest_path
        
        self.times = times
        
        self.bpm = bpm
        
        self.bpm_comp = bpm_comp
        
        self.file_info = file_info
        
        self.flag = flag


    def orientationExtractor(self,folder):
    
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
    
    
    def repetitionExtractor(self,folder):
            
            """
            Extract whether files to be read are repeated acquisitions or not.
            
            Params:
                - inherited from class (see description at beginning)
                - folder: folder name where to check for information
                
            Return: repetition string (rep) ('_0' for first acquisition, '_1' for second... and so on)
    
            """ 
            
            for i in range(5):
                
                test_str = '_' + str(i)
    
                if test_str in folder:
                
                    rep = '_' + str(i)
                    
                    break
                    
                else:
                    
                    rep = '_0'
            
            return rep
        
        
    
    def infoExtraction(self, image_folder, image_path, image_file, study_index, patientID):
        
        """
        Extract time information from desired folder and file (trigger time, 
        heart rate and expected heart rate).
        
        Params:
            
            - inherited from class (see class description at the beginning)
            - image_folder: folder where DICOM image is located
            - image_path: directory where to look for DICOM image file
            - image_file: DICOM file from where to extract the time information
            - study_index: number of study
            - patientID: patient code
        
        Returns:
            
            - if flag = 'save' --> updated time information on trigger time, 
            heart rate and expected heart rate, saved as TXT files
            
            - if flag = 'load' --> previously saved time information on 
            trigger time, heart rate and expected heart rate
        
        """
        
        orientation = self.orientationExtractor(image_folder)
                                
        rep = self.repetitionExtractor(image_folder)
   
        dcm = pydicom.read_file(image_path + image_file)
        
        time = float(dcm.TriggerTime)

        heart_rate = float(dcm.HeartRate)
        
        if dcm.Manufacturer == 'GE MEDICAL SYSTEMS':
            
            compute = round(60/((time/1000)*float(dcm[0x18, 0x1090])))
              
        else:
        
            compute = round(60/((time/1000)*40))

        self.bpm_comp.append(compute)
        
        self.times.append(time)
        
        self.bpm.append(heart_rate)

        self.file_info.append(self.studies[study_index] + '_' + patientID + '_' + orientation + rep)
    
    
    def existenceChecker(self, path, flag = 'folder'):
    
        """
        Checks that folder or file introduced exists.
        
        Params:
            
            - inherited from class (see class description at the beginning)
            
            - path: path to check if exists
            
            - flag: if 'folder' checks for existence of a folder // if 'file' ckecks for existence of a file
        
        Returns:
            
            - exists: flag telling if file or folder exists or not
        
        """
        
        if (flag == 'folder') or (flag == 'Folder') or (flag == 'FOLDER'):
        
            exists = os.path.isdir(path)
        
        elif (flag == 'file') or (flag == 'File') or (flag == 'FILE'):
            
            exists = os.path.isfile(path)
            
        
        else:
            
            print('Wrong flag introduced. Please introduce "folder" to check for folder existence or "file" to check for file existence')
    
            exists = None
        
        return exists
        
    
    def infoSaving(self):
        
        
        """
        Save time information (file information, trigger times, heart rate and 
        computed heart rate) in TXT files in a given folder.
        
        Params:
            
            - inherited from class (see class description at the beginning)
        
        Returns:
            
            - saved TXT files (one for file information, one for trigger times,
              one for heart rate and one for computed heart rate)
        
        """
        
        # Check that destination folder exists
        
        ex = self.existenceChecker(self.dest_path, 'folder')
        
        if ex:
            
            time_info_file = 'time_info_files.txt'
            
            time_info_times = 'time_info_times.txt'
            
            time_info_bpm = 'time_info_bpm.txt'
            
            time_info_comp_bpm = 'time_info_computed_bpm.txt'
            
            time_info_file_ex = self.existenceChecker(self.dest_path + time_info_file, 'file')
            
            time_info_times_ex = self.existenceChecker(self.dest_path + time_info_times, 'file')
            
            time_info_bpm_ex = self.existenceChecker(self.dest_path + time_info_bpm, 'file')
            
            time_info_comp_bpm_ex = self.existenceChecker(self.dest_path + time_info_comp_bpm, 'file')
            
            if time_info_file_ex:
                
                o = input('File {} already exists in folder. Do you want to overwrite it? [y/n]:'.format(time_info_file_ex))
                
                if (o == 'y') or (o == 'Y'):
                    
                    np.savetxt(self.dest_path + time_info_file, np.array(self.file_info), fmt = '%s') # TXT with file info
                
                else:
                    
                    print('Operation terminated by user\n')
            
            else:
                
                np.savetxt(self.dest_path + time_info_file, np.array(self.file_info), fmt = '%s') # TXT with file info
            
            if time_info_times_ex:
                
                o = input('File {} already exists in folder. Do you want to overwrite it? [y/n]:'.format(time_info_times_ex))
                
                if (o == 'y') or (o == 'Y'):
                    
                    np.savetxt(self.dest_path + time_info_times, np.array(self.times)) # TXT with file info
                
                else:
                    
                    print('Operation terminated by user\n')
            
            else:
                
                np.savetxt(self.dest_path + time_info_times, np.array(self.times)) # TXT with time info
                    
            if time_info_bpm_ex:
                
                o = input('File {} already exists in folder. Do you want to overwrite it? [y/n]:'.format(time_info_bpm_ex))
                
                if (o == 'y') or (o == 'Y'):
                    
                    np.savetxt(self.dest_path + time_info_bpm, np.array(self.bpm)) # TXT with file info
                
                else:
                    
                    print('Operation terminated by user\n')
                
            else:
                
                np.savetxt(self.dest_path + time_info_bpm, np.array(self.bpm)) # TXT with heart rate info
            
            if time_info_comp_bpm_ex:
                
                o = input('File {} already exists in folder. Do you want to overwrite it? [y/n]:'.format(time_info_comp_bpm_ex))
                
                if (o == 'y') or (o == 'Y'):
                    
                    np.savetxt(self.dest_path + time_info_comp_bpm, np.array(self.bpm_comp)) # TXT with file info
                
                else:
                    
                    print('Operation terminated by user\n')
            
            
            else:
 
                np.savetxt(self.dest_path + time_info_comp_bpm, np.array(self.bpm_comp)) # TXT with computed heart rate info
            
        else:
            
            print('Destination folder does not exist. Please create it\n')
            
    
    def infoLoading(self):
        
        """
        Load information on time from TXT files saved in a given folder.
        
        Params:
            
           - inherited from class (see class description at the beginning)
           
          
        Returns
        
            - info: array with information from analyzed images
            
            - times: array with time resolutions
            
            - bpm: array with heart rate
            
            - bpm_comp: array with computed heart rate values from time resolutions
        
        """

        # Check if folder exists

        ex = self.existenceChecker(self.paths[0], 'folder')

        if ex:
            
            txt_files = os.listdir(self.paths[0])
            
            for txt_file in txt_files: # Go through all text files
                
                if txt_file[-3:] == 'txt':
                    
                    # Access only TXT files
                    
                    if 'files' in txt_file:
            
                        file_info = np.loadtxt(self.paths[0] + txt_file, dtype = str)
                    
                    elif 'times' in txt_file:
                        
                        times = np.loadtxt(self.paths[0] + txt_file)
                    
                    elif ('bpm' in txt_file) and (not('computed' in txt_file)):
                        
                        bpm = np.loadtxt(self.paths[0] + txt_file)
                    
                    elif ('bpm' in txt_file) and ('computed' in txt_file):
                        
                        bpm_comp = np.loadtxt(self.paths[0] + txt_file)
  
            return file_info, times, bpm, bpm_comp
            
        else:
            
            print('Loading folder does not exist. Please create it\n')
            
            

    def __main__(self):
        
        if (self.flag == 'save') or (self.flag == 'Save') or (self.flag == 'SAVE'):
        
            print('\nExtracting time information from different studies\n')
        
            for i in range(len(self.paths)):
                
                if i != 2:
                    
                    patients = os.listdir(self.paths[i])
                    
                    # Access studies that are not heroic
    
                    for patient_folder in patients:
        
                        patientID = patient_folder[0:11] # For saving patient ID information
                        
                        # Dive into patient folders
       
                        patient_files = os.listdir(path + patient_folder + '/')
                        
                        for image_folder in patient_files:
                            
                            if self.folder_keys[i][0] in image_folder:
                            
                                image_path = path + patient_folder + '/' + image_folder + '/' # Image folder
                                
                                image_files = sorted(os.listdir(image_path))
                                
                                if i == 1: # For study of CKD2
                                    
                                    searched_file = patient_folder + '_IM_2.dcm'
                    
                                    if searched_file in image_files:
                                        
                                        self.infoExtraction(image_folder, image_path, searched_file, i, patientID)
                                        
                                
                                else: # For CKD1 and Extra studies
                
                                    self.infoExtraction(image_folder, image_path, image_files[1], i, patientID)
                    
                else:
                    
                    # Access GE files in heroic studies
    
                    individuals = os.listdir(self.paths[i])
                    
                    for individual in individuals: # Go through healthy volunteers and pilot patients
        
                        if individual != 'Clean_data':
                    
                            individual_path = self.paths[i] + individual + '/' # Healthy or patient
                        
                            vendors = os.listdir(individual_path) # Vendor: GE or Siemens
                        
                            for vendor in vendors: # Go through the different vendors
                                
                                vendor_path = individual_path + vendor + '/' # Vendor folder
                                
                                patients = os.listdir(vendor_path) # List of patients
                                
                                for patient in patients: # Go through all patients
                                    
                                    patient_path = vendor_path + patient + '/' # Patient folder
                                    
                                    patient_files = os.listdir(patient_path) # Patient images
                                    
                                    if 'GE' in vendor: # Access GE files
                        
                                        for image_folder in patient_files: # Access folder with DCM images
                                            
                                            if (self.keys[i][0] in image_folder) or (self.keys[i][1] in image_folder): # Access only PC-MRI
                    
                                                image_path = patient_path + image_folder + '/'
                                                
                                                image_files = sorted(os.listdir(image_path)) # Image files
                                                
                                                self.infoExtraction(image_folder, image_path, image_files[1], i, patientID)
    
                print('Information was extracted successfully!\nSaving information into TXT files in {}'.format(self.dest_path))
                
                self.infoSaving()
                
                print('Information saved successfully!\n')
                
                plot = True
                
                save = True
                
                # Remove defective points from BPM arrays
                
                remove = list(np.where(np.array(self.bpm_comp) <  min(self.bpm)))
                
                bpm_array = np.delete(np.array(self.bpm), remove[0], axis = None)
                
                bpm_comp_array = np.delete(np.array(self.bpm_comp), remove[0], axis = None)
                
                c, mse, r2 = linear_regression_test(bpm_comp_array, bpm_array, plot, save, dest_path, filename = 'regression_plot.png')
                
                return c, mse, r2
        
        elif (self.flag == 'load') or (self.flag == 'Load') or (self.flag == 'LOAD'):
            
            print('Loading time information from TXT files in {}'.format(self.paths[0]))
            
            file_info, times, bpm, bpm_comp = self.infoLoading()
            
            print('Information loaded successfully!\n')
            
            return file_info, times, bpm, bpm_comp
        
        else:
            
            print('Wrong flag given. Please give a valid flag (save/load)\n')




study1 = 'CKD1'

study2 = 'CKD2'

study3 = 'hero'

study4 = 'extr'

key = ['QFLOW']

path = '/home/andres/Documents/_Data/CKD_Part1/' # Folder with CKD Part 1 study info

path2 = '/home/andres/Documents/_Data/CKD_Part2/1_Raw_data/' # Folder with CKD Part 2 study info

heroic_path = '/home/andres/Documents/_Data/Heroic/' # Folder with Heroic files

extra_path = '/home/andres/Documents/_Data/Extra/' # Folder with Extra files

dest_path = '/home/andres/Documents/_Data/Time_info/' # Folder with time information

key_hero_ge = ['PC', 'QFLOW']

paths = [path, path2, heroic_path, extra_path]

folder_keys = [key, key, key_hero_ge, key_hero_ge[0]]

studies = [study1, study2, study3, study4]


times = [20, 23, 24, 24, 23] # List with trigger times

bpm = [76, 64, 60, 62, 68] # List with real cardiac frequencies

file_info = ['extr_20181213_si_0', 'extr_20181213_si_1', 'extr_20181215_dx_0', 'extr_20181215_si_0', 'extr_20181215_si_1'] # List with file information

comp_bpm = np.round(60/((np.array(times)/1000)*40)) 

comp_bpm_list = list(comp_bpm) # List with computed cardiac frequency from trigger times

#timeExtractLoad = TimeAnalyzer([dest_path], studies, folder_keys, dest_path, times, bpm, comp_bpm, file_info, 'load')

#file_info, times, bpm, bpm_comp = timeExtractLoad.__main__()

timeExtractSave = TimeAnalyzer(paths, studies, folder_keys, dest_path, times, bpm, comp_bpm_list, file_info, 'save')

ce, mse, r2 = timeExtractSave.__main__()


# To execute from terminal:

# paths = sys.argv[1]
# studies = sys.argv[2] 
# folder_keys = sys.argv[3]
# dest_path = sys.argv[4] 
# times = sys.argv[5]
# bpm = sys.argv[6]
# comp_bpm_list = sys.argv[7]
# file_info = sys.argv[8]
     


# if __name__ == "__main__":  