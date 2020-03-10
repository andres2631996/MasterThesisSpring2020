#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:58:47 2020

@author: andres
"""

import numpy as np

import os

import pandas as pd

from flowInformation import FlowInfo

import random

from sklearn.model_selection import StratifiedKFold

import itertools


class StratKFold:
    
    """
    Provide list of study + patient folder where to look for images in a 
    stratified k-fold cross validation.
    
    Params:
        
        - folders: list of paths where to look for information
        
        - test_folder: path where to extract test images
        
        - test_file: Excel file with test flow measurements
        
        - test_raw_folder: folder with processed test images
        
        - studies: list of studies where to look for information
        
        - k: number of folds desired
        
        - rep: flag stating whether to artificially repeat patients from 
               minority studies
        
    
    Returns: 
        
        - val_lists: list of study + patient folder where to look for images in a 
        stratified k-fold cross validation
        
        - test_images: images for testing from Excel file
        
        - test_flows: flow values for testing, from Excel file
    
    
    """
    
    def __init__(self, folders, test_folder, test_file, test_raw_path, studies, k, rep):
        
        self.folders = folders
        
        self.studies = studies
        
        self.test_folder = test_folder
        
        self.test_file = test_file
        
        self.test_raw_folder = test_raw_path
        
        self.k = k
        
        self.rep = rep
    
    

    def StratifiedKFoldPatients(self, patients, labels):
        
        """
        Apply patient stratification on given patients and given labels. Labels are
        based on mean arterial flow quantification from the patient.
        
        Params:
            
            - k: number of folds to apply
            
            - patients: list of patients to stratify
            
            - labels: list of labels corresponding to measured patient flow
        
        Returns:
            
            - val_lists: lists with patients used for validation in each fold
        
        """
        
        
    
        # Now decide fold information so that patients from the same group fall into different folds
            
        skf = StratifiedKFold(n_splits = self.k, shuffle = True, random_state = 1)
    
        skf.get_n_splits(patients, labels)
        
        val_lists = []
        
        cont = 0
        
        for train_index, test_index in skf.split(patients, labels):
    
            val_lists.append(patients[test_index].tolist())
            
            random.shuffle(val_lists[cont]) # Shuffle patients inside
            
            cont += 1
            
        return val_lists
    
    
    
    
    def flowFromExcel(self):
        
        """
        Extract flow information from Excel file, given the path with
        the Excel file, the name of the Excel file and the path of the raw images of
        the study (usually CKD1), arranged into patients.
        
        Params:
            
            - excel_path: folder where to look for Excel file
            
            - excel_file: filename with flow information
            
            - path_ckd1: folder where to find CKD1 images, arranged into patients
        
        Returns:
            
            - patients_final_ckd1: list of CKD1 patients with images and flow information
            
            - flow_final_ckd1: flow information from CKD1 study
            
        """
        
        # Extract measured patients from CKD1
    
        available_patients = sorted(os.listdir(self.test_raw_folder))
        
        # Delete defective patients
        
        ind_def = [i for i, s in enumerate(available_patients) if 'CKD015' in s]
        
        available_patients = np.array(available_patients)
        
        available_patients = np.delete(available_patients, ind_def)
        
        available_patients = [z[:6] for z in available_patients]
        
        df = pd.read_excel(self.test_folder + self.test_file) # can also index sheet by name or fetch all sheets
        
        flow_ckd1 = np.array(df['Mean_arterial_flow'].tolist()) # List with flow values
        
        patients_ckd1 = df['Subject_ID'].tolist() # List with patients
        
        test_ckd1 = df['MR_Visit'].tolist() # List with patients
        
        orient_ckd1 = np.array(df['left_right_kidney'].tolist()) # List with patients
        
        ind_left = np.where(orient_ckd1 == 'left')
        
        ind_right = np.where(orient_ckd1 == 'right')
        
        orient_final = np.copy(orient_ckd1)
        
        orient_final[ind_left] = 'si'
        
        orient_final[ind_right] = 'dx'
        
        patient_final = []
        
        for i in range(len(patients_ckd1)):
            
            patient_final.append('CKD1_'+ patients_ckd1[i] + '_' + test_ckd1[i] + '_' + orient_final[i])
        
#        ind_defective = np.where(patients_ckd1 == 'CKD015')
#        
#        patients_ckd1 = np.delete(patients_ckd1, ind_defective) 
#        
#        flow_ckd1 = np.delete(flow_ckd1, ind_defective) 
#    
#        for patient in available_patients:
#            
#            patients_final_ckd1.append('CKD1_' + patient)
#            
#            ind_patient = np.where(patients_ckd1 == patient) # Indices per patient where to focus
#            
#            flow_final_ckd1.append(np.mean(flow_ckd1[ind_patient]))
        
        return patient_final, flow_ckd1
    
    

    def unison_shuffled_copies(self, a, b):
        
        """
        Shuffle two arrays or lists in the same way.
        
        Params:
            
            - a and b: lists or 1D arrays to shuffle in the same way
            
        Returns:
            
            - shuffled lists or 1D arrays in the same way
        
        """
        
        assert len(a) == len(b)
        
        p = np.random.permutation(len(a))
        
        return a[p], b[p]

    def flowFromMat(self, path, study):
        
        """
        Obtain patients and flow information from flow measurements as TXT files 
        from given study in given folder.
        
        Params:
            
            - study: CKD2/Hero/Extr
            
            - ckd2path: folder with CKD2 flow measurements
        
        
        Returns:
            
            - patients_final: list of CKD1 patients with images and flow information
            
            - flow_final: flow information from CKD1 study
        
        """
        
        patients = sorted(os.listdir(path))
    
        flow_files = patients.copy()
        
        if study == 'CKD2':
        
            patients = [z[5:16] for z in patients]
        
        elif study == 'Hero':
            
            p = []
            
            for z in patients:
                
                ind_ = [pos for pos, char in enumerate(z) if char == '_']
                
                if 'venc' in z:
                    
                    p.append(z[:ind_[-4]]) 
    
                else:
    
                    p.append(z[:ind_[-3]]) 
            
            patients = p.copy()
        
        elif study == 'Extr':
            
            patients = [z[:17] for z in patients]
        
        patients = np.array(patients)
        
        patients_unique = sorted(np.unique(patients))
        
        flow_final = []
        
        patients_final = []
        
        for patient in patients_unique:
            
            if patient[:6] != 'CKD007':
                
                if study == 'CKD2':
            
                    patients_final.append('_' + study.lower() + '/' + patient[:6] + '_MRI3/')
                
                else:
                    
                    patients_final.append(str('_' + patient[:4].lower() + '/' + patient[5:] + '/'))
                
                ind_patient = np.array(np.where(patients == patient))[0] # Indices per patient where to focus
                
                flow_patient = []
                
                for ind in ind_patient:
                    
                    load_info_mat = FlowInfo(study, path, None, None, 'load', True, flow_files[ind])
                    
                    _, _, _ , _, _, _ , net_flow, _, _ = load_info_mat.__main__()
                    
                    flow_patient.append(abs(np.mean(net_flow)))
                
                flow_final.append(np.mean(np.array(flow_patient)))
        
        return patients_final, flow_final


    def stratification(self, patients, flows):
    
        """
        Provide k groups of patients divided according to their flow.
        
        Params:
            
            - k: number of groups to divide patients into
            
            - patients: all patient names
            
            - flows: all flow measurements
        
        
        Returns:
            
            - final_patients: list of ordered patients in flow
            
            - final_labels: list with labels according to patient measured flow
    
        """   
    
        flow_array = np.array(flows)
    
        # Normalize flow array
        
        patient_array = np.array(patients)
        
        ind_sorted = np.argsort(flow_array)
        
        patient_sorted = (patient_array[ind_sorted]).tolist()
        
        patient_groups = [] # List with patients sorted into k groups
        
        patient_labels = [] # List of lists with labels indicating where the patient is located
        
        # Patients are now divided into k groups depending on their mean flow
    
        for group in range(self.k):
            
            if group < (self.k-1):
                
                patient_groups.append(patient_sorted[round(group*len(patient_sorted)/self.k):round((group + 1)/self.k*len(patient_sorted))])
                
                patient_labels.append([group]*len(patient_sorted[round(group*len(patient_sorted)/self.k):round((group + 1)/self.k*len(patient_sorted))]))
        
            else:
                
                patient_groups.append(patient_sorted[round((self.k - 1)*len(patient_sorted)/self.k):])
                
                patient_labels.append([group]*len(patient_sorted[round((self.k - 1)*len(patient_sorted)/self.k):]))
            
            random.shuffle(patient_groups[group]) # Element shuffling inside list
            
        
        final_patients = np.array(list(itertools.chain.from_iterable(patient_groups)))  
        
        final_labels = np.array(list(itertools.chain.from_iterable(patient_labels)))
        
        final_patients, final_labels = self.unison_shuffled_copies(final_patients, final_labels)
    
        return final_patients, final_labels        
            


    def __main__(self):
        
        test_images, test_flows = self.flowFromExcel() # Get test results
        
        all_flows = []
        
        all_patients = []
        
        for study, path in zip(self.studies, self.folders):
            
            patients, flow = self.flowFromMat(path, study)
                
            all_flows.append(flow)
                
            all_patients.append(patients)
        
        all_flows = list(itertools.chain.from_iterable(all_flows))

        all_patients = list(itertools.chain.from_iterable(all_patients))
        
        patients, labels = self.stratification(all_patients, all_flows)
           
        # Sort patients and flows according to their value
            
        val_lists = self.StratifiedKFoldPatients(patients, labels)
        
        # Repeat patients in each fold from minority studies
        
        if self.rep:
            
            for val_list in val_lists:
                
                ind_hero = [i for i,s in enumerate(val_list) if 'hero' in s]
                
                ind_extr = [i for i,s in enumerate(val_list) if 'extr' in s]
                
                if len(ind_hero) != 0:
                    
                    for ind_h in ind_hero:
                        
                        for i in range(3):
                            
                            val_list.append(val_list[ind_h])
                
                
                if len(ind_extr) != 0:
                    
                    for ind_e in ind_extr:
                        
                        for i in range(5):
                            
                            val_list.append(val_list[ind_e])
        
                random.shuffle(val_list)
        
        return val_lists, test_images, test_flows
            
    
# Test code


rep = True # Factor allowing for repetition of patients from minority studies (Heroic and Extra)

# Import information of EXCEL file with flow measurements of CKD1

excel_path = '/home/andres/Documents/_Data/CKD_Part1/'

excel_file = 'CKD_QFlow_results.xlsx' 

raw_path_ckd1 = '/home/andres/Documents/_Data/_Patients/_Raw/_ckd1/'

# CKD flow measurements in CKD Part 2

raw_path_ckd2 = '/home/andres/Documents/_Data/CKD_Part2/4_Flow/'

study2 = 'CKD2'

raw_path_hero = '/home/andres/Documents/_Data/Heroic/_Flow/'

studyh = 'Hero'

raw_path_extr = '/home/andres/Documents/_Data/Extra/_Flow/'

studye = 'Extr'

k = 5
    

strKFolds = StratKFold([raw_path_ckd2, raw_path_hero, raw_path_extr], excel_path, 
                            excel_file, raw_path_ckd1,[study2, studyh, studye], k, rep)

val_lists, test_img, test_flow = strKFolds.__main__()

# To run in terminal:

# folder_paths = sys.argv[1]
# excel_path = sys.argv[2] 
# excel_file = sys.argv[3]
# test_img_path = sys.argv[4] 
# studies = sys.argv[5]
# k = sys.argv[6]
# rep = sys.argv[7]
    

# if __name__ == "__main__": 

