#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:35:58 2020

@author: andres
"""

import numpy as np

import scipy

import os

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm



# Code to deal with statistics in data post-processing

def t_test(result, reference):
    
    """
    Compute the t statistic and p-value given the flow results from the neural
    network and the reference results from Segment.
    
    Params:
        - result: 1D array with flow results from neural network segmentation
        - reference: 1D array with reference flow results from software Segment
    
    Returns:
        - t_statistic and p_value
    
    """
    
    # Check that result and reference are 1D and that they have the same length
    
    print('\nChecking that result and reference are 1D and that they have the same length\n')
    
    if (len(result.shape) == 1) and (len(reference.shape) == 1):
        
        if len(result) == len(reference):
            
            print('Performing t test\n')
    
            t_stat, p_value = scipy.stats.ttest_ind(result, reference)
            
            print('t test completed successfully!\n')
            
            print('t statistic: {} // p value: {}'.format(t_stat, p_value))
            
            return t_stat, p_value
        
        else:
            
            print('Result and reference vectors do not have the same length. Please input them so that they have the same length')
        
    else:
        
        print('Result or reference vectors are not 1D. Please reformat them to be 1D')
        
        


def wilcoxon_test(result, reference):
    
    """
    Compute the sum of rank differences and p-value given the flow results 
    from the neural network and the reference results from Segment.
    
    Params:
        - result: 1D array with flow results from neural network segmentation
        - reference: 1D array with reference flow results from software Segment
    
    Returns:
        - sum and p_value
    
    """
    
    print('\nChecking that result and reference are 1D and that they have the same length\n')
    
    if (len(result.shape) == 1) and (len(reference.shape) == 1):
        
        if len(result) == len(reference):
            
            print('Performing Wilcoxon test\n')
            
            s, p_value = scipy.stats.wilcoxon(result, reference)
            
            print('Wilcoxon test completed successfully!\n')
            
            print('Sum of rank differences: {} // p value: {}'.format(s, p_value))
    
            return s, p_value
        
        else:
            
            print('Result and reference vectors do not have the same length. Please input them so that they have the same length')
        
    else:
        
        print('Result or reference vectors are not 1D. Please reformat them to be 1D')
        
        

def figure_saving(dest_path, filename, figure):
    
    """
    Saves figure in specified folder with a given file name
    
    Params:
        
        - dest_path: destination folder
        
        - filename: given filename (must be .png)
        
        - figure: given figure from Matplotlib
    
    """
    
    print('Checking that a destination path has been given\n')
    
    if dest_path is not None:
        
        if filename[-3:] == 'png' or filename[-3:] == 'PNG':
                        
            print('Saving figure as PNG in: {}'.format(dest_path))
            
            # Check that folder exists. Otherwise, create it
            
            print('\nChecking if folder exists\n')
            
            exist = os.path.isdir(dest_path)
            
            if not exist:
                
                os.makedirs(dest_path)
                
                print('Non existing folder. Created\n')
                
            # Checking if file exists
            
            files = os.listdir(dest_path)
    
            if filename in files:
                
                inp = input('File already exists. Do you want to overwrite or rename or abort? [o/r/a]:')
                
                if (inp == 'o') or (inp == 'O'):
                    
                    figure.savefig(dest_path + filename)
                
                elif (inp == 'r') or (inp == 'R'):
                    
                    cont = 0 # Filename counter
                    
                    while (filename in files):
                        
                        filename = filename[:-4] + '_' + str(cont) + '.png'
                        
                        cont += 1
                    
                    figure.savefig(dest_path + filename)
                    
                    print('Figure successfully saved as PNG in {} after file renaming'.format(dest_path))
                
                else:
                    
                    print('Operation aborted\n')
            else:
        
                figure.savefig(dest_path + filename)
            
                print('Figure successfully saved as PNG in {}'.format(dest_path))
            
        else:
            
            print('Filename given was not PNG. Please specify a PNG filename')
    
    else:
        
        print('Tried to save figure as PNG, but folder has not been specified')



def linear_regression_test(result, reference, plotting = True, save = False, dest_path = os.getcwd() + '/', filename = 'regression_plot.png'):
    
    """
    Compute a scatter plot with points and with linear regression equation
    given the flow results from the neural network and the reference results 
    from Segment.
    
    Params:
        - result: 1D array with flow results from neural network segmentation
        - reference: 1D array with reference flow results from software Segment
        - plotting: flag to state whether the user wants to get a plot of the result or not (default True)
        - save: flag indicating whether to save linear regression plot as .png file or not (default False)
        - dest_path: folder where to save the linear regression plot if save is True (default current folder)
        - filename: file name of regression plot (PNG file) (default: 'regression_plot.png')
    
    Returns:
        - linear regression plot and linear regression parameters
        - Return also squared error and R2
    
    """
    
    print('\nChecking that result and reference are 1D and that they have the same length\n')
    
    if (len(result.shape) == 1) and (len(reference.shape) == 1):
        
        if len(result) == len(reference):
            
            print('Computing linear regression model\n')
    
            # Create linear regression object
            
            regr = linear_model.LinearRegression()
            
            # Train the model
            
            # Add a column of ones to the reference
            
            ref_aux = np.ones((len(reference), 2))
            
            ref_aux[:,0] = reference
            
            regr.fit(ref_aux, result)
    
            # Make predictions using the reference 
            
            pred = regr.predict(ref_aux)
            
            print('Linear regression model completed successfully!\n')
            
            print('Resulting coefficients: {}\n'.format(regr.coef_))

            # The mean squared error
            
            print('Mean squared error: {}\n'.format(mean_squared_error(result, pred)))
            
            # The coefficient of determination: 1 is perfect prediction
            
            print('Coefficient of determination: {}\n'.format(r2_score(result, pred)))
            
            if plotting:
                
                fig = plt.figure()
                
                # Points
                
                plt.scatter(reference, result, color = 'black', label = 'Measured points')
                
                # Regression line
                
                plt.plot(reference, pred, color='blue', linewidth=3, label = 'Regression line')
                
                plt.title('Linear regression plot')
                
                plt.xlabel('Reference values')
                
                plt.ylabel('Computed values')
                
                plt.legend()
                
                plt.xticks(())
                
                plt.yticks(())
                
                plt.show()
                
                if save:
                    
                    figure_saving(dest_path, filename, fig)
                    
            return regr.coef_, mean_squared_error(result, pred), r2_score(result, pred)
            
        else:
            
            print('Result and reference vectors do not have the same length. Please input them so that they have the same length')
        
    else:
        
        print('Result or reference vectors are not 1D. Please reformat them to be 1D')
    



def bland_altman_plot(result, reference, save = False, dest_path = os.getcwd() + '/', filename = 'bland_altman_plot.png'):
    
    """
    Compute the Bland-Altman plot for the flow results from the neural network 
    and the reference results from Segment.
    
    Params:
        - result: 1D array with flow results from neural network segmentation
        - reference: 1D array with reference flow results from software Segment
        - save: flag indicating whether to save linear regression plot as .png file or not (default False)
        - dest_path: folder where to save the linear regression plot if save is True (default current folder)
        - filename: file name of regression plot (PNG file) (default bland_altman_plot.png)   
        
    
    
    """
    
    print('\nChecking that result and reference are 1D and that they have the same length\n')
    
    if (len(result.shape) == 1) and (len(reference.shape) == 1):
        
        if len(result) == len(reference):
            
            print('Computing Bland-Altman plot\n')
            
            f, ax = plt.subplots(1)
            
            sm.graphics.mean_diff_plot(result, reference, ax = ax)
            
            plt.title('Bland-Altman plot')
                
            plt.xlabel('Average values')
            
            plt.ylabel('Difference values')
    
            plt.show()
    
            if save:
                    
                figure_saving(dest_path, filename, f)
            
        else:
            
            print('Result and reference vectors do not have the same length. Please input them so that they have the same length')
        
    else:
        
        print('Result or reference vectors are not 1D. Please reformat them to be 1D')
    
    
def correlation(result, reference):
 
    """
    Compute Pearson correlation coefficient (r) between network result and ground-truth
    
    Params:
    
        - result: network result
        
        - reference: ground-truth
        
    Returns:
    
        - r: Pearson correlation coefficient

    """
    
    r = np.corrcoef(result, reference)[0,1]
    
    return r