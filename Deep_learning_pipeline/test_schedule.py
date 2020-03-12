#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:14:53 2020

@author: andres
"""

import numpy as np

import schedule

import time

import os

import shutil

import runpy


#import crossValidation


#def job2():
    
run_path = '/home/andres/Documents/_Code/runs/'

param_files = sorted(os.listdir(run_path))

while len(param_files) != 0:

    
    # Take the first file in the runs folder and rename it to "params.py"
    
    os.rename(run_path + param_files[0], run_path + 'params.py')
    
    # Copy the file into the code folder and replace the last "params.py" file

    shutil.copyfile(run_path + 'params.py', run_path[:-5] + 'params.py')
    
    # Remove the "params.py" file from the "runs" folder
    
    os.remove(run_path + 'params.py')
    
    # Execute the cross-validation code
    
    runpy.run_path(run_path[:-5] + 'crossValidation.py')
        
        

#schedule.every().day.at("17:24").do(job2)
#
#
#
#
#while True:
#    
#    schedule.run_pending()
#    
#    time.sleep(1)