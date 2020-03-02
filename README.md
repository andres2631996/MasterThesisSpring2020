# MasterThesisSpring2020
Code for my Master Thesis on spring 2020 about automatic kidney perfusion analysis in 2D PC-MRI with deep learning

The data is structured as a parent folder called "/Patients/". Then it has 3 subfolders: "/Raw/" with raw data, "/Prep/" with pre-processed data without cropping and "/Crop/" with pre-processed and cropped data

Code files:

"matContourToBinary.py" reads a .mat file with a ROI contour and saves it as a binary mask in VTK format
If the measurement comes from a cropping, it is zero-padded with an internal function
If the .mat file contains a magnitude image corrected for eddy currents, it allows to save it in a given path too
One must specify in the terminal: study path_of_mat_file mat_filename whether_to_save_or_not_contour_image(True/False) destination_path
path_for_eddy_corrected_images
Requires to have installed the following libraries: Numpy, Scipy, MatplotLib and VTK


"vtkBinaryToContour.py" reads a VTK file with a binary or quasi-binary mask and saves it as TXT file.
If array is 3D, the contour of each slice/frame is separated by [-1,-1] as delimiter in the TXT file
An extra TXT file is saved with information on origin and spacing. 3 first rows for origin 
(originX, originY, originZ) and 3 last rows for spacing (spacingX, spacingY, spacingZ)
One must specify in the terminal: vtk_filename path_of_vtk_file destination_path
Requires to have installed the following libraries: Numpy, Scikit-Image, Matplotlib, VTK


"txtContour2array.py" reads a TXT file with contours and provides a list with the contours in an array per frame/slice (if 3D).
Also compatible in 2D.
It returns the arrays both in spatial and in pixel coordinates.
One must specify in the terminal: path_of_txt_file txt_filename destination_path
Requires to have installed the following libraries: Numpy


"rawFileReader.py" takes as input a study name, a path to start looking for the images of the study, a key to look for the 2D QFLOW images
of the study and a folder where to save the images and it arranges the images as 3D arrays in VTK files in the specified destination.
One must specify in the terminal: study path_to_find_images destination_path key_to_search_2DQFLOW
Requires to have installed the following libraries: NumPy, VTK and Pydicom 


"maskAlignment.py" takes as input a folder with raw files, a folder with cropped masks from previous measurements and a folder where to save 
the resulting images. With these, it zero-pads the cropped images to the dimensions of the raw files, registers the masks to the raw files and
refines the resukts from the registration. Final results are saved as VTK files in the specified folder.
One must specify in the terminal: folder_with_raw_files folder_with_masks destination_folder number_iterations learning_rate allowForPatchRegistration
				  patch_init_x patch_end_x patch_init_y patch_end_y
Requires to have installed: NumPy, VTK and SimpleITK


"timeInfoExtractor.py" extracts time information from DICOM files (non Siemens) and saves it in TXT files in a given location.
Alternatively, it also allows for loading TXT files with previously saved information.
If flag = 'save', the code additionally executes a linear regression analysis of the extracted heart rate and the computed heart rate from the 
extracted time resolutions, providing with linear regression coefficients, mean-squared error and R2 coefficient.
One must specify in the terminal: list_of_study_paths study_list key_list destination_path initial_time_array initial_bpm_array initial_computed_bpm_array
initial_file_information_array flag('save'/'load')
Requires to have installed: NumPy, Pydicom and Scikit-learn


"patientOrganizer.py" enters a hierarchy of folders with disarranged VTK files and arranges those files as study --> patients hierarchy,
allowing later on for cross-validation in terms of patients.
One must specify in the terminal: parent_folder_disarranged_files parent_folder_arranged_files
Requires to have installed: numpy, shutil


"filePreprocessing.py" preprocesses VTK raw files arranged into studies and patients in a specified folder, saving the pre-processed files into studies and
patients in a destination folder. Pre-processing consists on setting all files to the same matrix size and on normalizing between -1 and +1 the
magnitude images. Pre-processed files are saved as VTK in the destination folder. 
One must specify in the terminal: folder_with_raw_files destination_folder matrixSize_to_set_all_files
Requires to have installed: NumPy, VTK, os, sys, time


"patientStratification.py" gets information on mean arterial flow from patients in different studies and prepares a stratified list of k lists
of stratified patients according to their mean arterial flow, obtained either from Excel files or .mat files from Segment
One must specify in the terminal: list_folders_for_training/validation excel_file_path excel_filename path_with_ckd1Images studiesForTraining/Validation
				  k repeat/dont_repeat_artificially_data_from_minority_studies
Requires to have installed: NumPy, os, pandas, random, scikit-learn, itertools and FlowInformation.py script (see info on README_postprocessing)
