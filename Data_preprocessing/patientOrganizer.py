#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:08:30 2020

@author: andres
"""

import os

import numpy as np

import shutil

import sys


class patientOrganizer:
    
    """
    Organizes a set of VTK files into patients. Images from same patient and 
    same study are saved in the same folder.
    
    Params:
        
        - start_path: folder where to look for disorganized data
        
        - end_path: folder where to save all arranged data
    
    Returns:
        
        - hierarchy of folders of study --> patients --> images where everything
          is properly arranged
    
    
    """
    
    def __init__(self, start_path, end_path):
        
        self.start_path = start_path
        
        self.end_path = end_path
    
    
    def __main__(self):

        modalities = os.listdir(self.start_path) # Modalities found (mag, mag_biasField, pha...)
        
        for modality in modalities:
        
            modality_path = self.start_path + modality + '/' # modality path
            
            studies = os.listdir(modality_path) # ckd1, ckd2, hero, extr...
            
            for study in studies: 
                
                study_path = modality_path + study + '/' # Study path
                
                studyID = study[5:]
                
                end_study_folder = self.end_path + '_' + studyID + '/'
                
                images = sorted(os.listdir(study_path)) # Images from same study and modality
                
                patient_list = []
                
                for image in images:
                    
                    # Extract all patient IDs
                    
                    list_info = image.split('_')
                    
                    if (studyID == 'ckd1') or (studyID == 'ckd2') or (studyID == 'extr'):
                    
                        patientID = list_info[1] + '_' + list_info[2] # Name for patient folder
                    
                    else: 
                        
                        if (list_info[2] == 'dx') or (list_info[2] == 'si'):
                            
                            patientID = list_info[1]
                        
                        else:
                            
                            patientID = list_info[1] + '_' + list_info[2] # Name for patient folder
                        
                    patient_list.append(patientID)
                
                patient_list_unique = list(set(patient_list)) # List with unique patients
                
                images_array = np.array(images)
                
                patient_list_array = np.array(patient_list)
                
                for patient in patient_list_unique:
                    
                    end_patient_folder = end_study_folder + patient + '/' # Final patient folder inside same study
                    
                    if not os.path.exists(end_patient_folder):
                        
                        # Non-existing folder. Create it
                        
                        os.makedirs(end_patient_folder[:-1])
                    
                    ind_patient = np.array(np.where(patient_list_array == patient))[0] # Index with images from same study, modality and patient
                    
                    files_found = images_array[ind_patient] # Files from same patient, study and modality
                    
                    for file in files_found:
                        
                        shutil.copy(study_path + file, end_patient_folder)
                        
                        
                
                
start_path = '/home/andres/Documents/_Data/_Raw_data/' # Folder where to start from

end_path = '/home/andres/Documents/_Data/_Patients/_Raw/' # Folder where to save everything   

patOrg = patientOrganizer(start_path, end_path)

patOrg.__main__()         


# To run in terminal:

# start_path = sys.argv[1]
# end_path = sys.argv[2] 
    


# if __name__ == "__main__":         