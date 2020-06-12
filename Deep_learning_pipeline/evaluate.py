#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:46:50 2020

@author: andres
"""

import numpy as np

import cv2

import torch

import vtk

from vtk import vtkStructuredPointsWriter, vtkStructuredPointsReader, vtkStructuredPoints

from vtk import vtkMetaImageReader, vtkImageAppend, VTK_BINARY

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import params

import time

import math

import matplotlib.pyplot as plt

import utilities


    
    
class Dice():
    
    """
    Compute Dice Coefficient as metrics.
    
    Params:
        
        - ground_truth: mask (PyTorch tensor)
        
        - prediction: network output (PyTorch tensor)
    
    
    Returns:
        
        - dice coefficient value (float)
    
    """
    
    def __init__(self, ground_truth, prediction):
        
        self.ground_truth = ground_truth
        
        self.prediction = prediction
        
    def online(self):
        
        numerator = 0
        
        denominator = 0
        
        if params.add3d > 0:
            
            self.ground_truth = self.ground_truth[:,:,:,params.add3d]
            
            self.prediction = self.prediction[:,:,:,params.add3d]
        
        numerator += float(torch.sum(torch.mul(self.ground_truth, self.prediction)))
        
        denominator += float(torch.add(torch.sum(self.ground_truth), torch.sum(self.prediction)))
        
        return ((2*abs(numerator))/(abs(denominator) + 0.000000001))
   


class Precision():
    
    """
    Compute precision value as metrics
    
    Params:
        
        - ground_truth: mask (PyTorch tensor)
        
        - prediction: network output (PyTorch tensor)
        
    Returns:
    
        - precision value (float)
    
    """
    
    def __init__(self, ground_truth, prediction):
        
        self.ground_truth = ground_truth
        
        self.prediction = prediction
    
    
    def online(self):
        
        numerator = 0
        
        denominator = 0
        
        numerator += float(torch.sum(torch.mul(self.ground_truth, self.prediction)))
        
        denominator += float(torch.sum(self.prediction))
        
        return (numerator/(denominator + 0.000000001))
    
    


class Recall():
    
    """
    Compute recall value as metrics
    
    Params:
        
        - ground_truth: mask (PyTorch tensor)
        
        - prediction: network output (PyTorch tensor)
        
    Returns:
    
        - recall value (float)
    
    """
    
    def __init__(self, ground_truth, prediction):
        
        self.ground_truth = ground_truth
        
        self.prediction = prediction
    
    
    def online(self):
        
        numerator = 0
        
        denominator = 0
        
        numerator += float(torch.sum(torch.mul(self.ground_truth, self.prediction)))
        
        denominator += float(torch.sum(self.ground_truth))
        
        return (numerator/(denominator + 0.000000001))
        




def evaluate(net, loader, iteration, key):
    
    """
    Evaluations done during cross-validation
    
    :param net: (PyTorch model)
        
    :param loader: (PyTorch dataloader, usually the validation dataloader)
        
    :param iteration: (int)
    
    :param: key: tells if evaluation is done for validation ("val") or for testing ("testing")

    :return: results: metric results (list of lists), both in validation and testing
    
    :return: raw_files: image filenames (list of str), only in testing mode
    
    :return: net_results: network predictions (list of arrays), only in testing mode
    
    :return: ground_truths: masks used as ground-truths (list of arrays), only in testing mode
    
    :return: names: image identifier extracted from the dataloader (list of str), only in testing mode
    
    
    """
    
    
    with torch.no_grad():
        
        params.test = True
        
        net.eval()  # This is necessary for the dropout layers to work

        batch_gpu_max = params.batch_GPU_max_inference
        
        batch_gpu = params.batch_GPU
        
        ram_batch_size = params.RAM_batch_size
        
        metrics = params.metrics
        
        dices = [] # List with evaluated Dice values
        
        precisions = [] # List with evaluated precisions
        
        recalls = [] # List with evaluated recalls
        
        results = [] # List with overall results: METRIC NAME, MEAN AND STD!!
        
        ground_truths = [] # List for saving ground truths during model testing
        
        net_results = [] # List for saving network results during model testing
        
        names = [] # List for saving filenames that are being evaluated during testing
        
        raw_files = [] # List for saving raw files to be evaluated in the testing process

        for X, Y, n in loader:
            
            # Extracts the data from dataloader and puts it into the network

            for i in range(min(math.ceil(ram_batch_size/batch_gpu_max), math.ceil(X.shape[0]/batch_gpu_max))):
                
                startIndiex = i*batch_gpu_max
                
                stopXIndex = min((i+1)*batch_gpu, X.shape[0])
                
                if len(Y) != 0:
                
                    stopYIndex = min((i+1)*batch_gpu, Y.shape[0])
                
                if params.three_D: # Full 2D+time volume (unused)
    
                    x_part = X[startIndiex:stopXIndex,:,:,:,:].cuda(non_blocking=True) #create a mini-batch of samples that fits on the GPU
                    
                    if len(Y) != 0:
            
                        y_part = Y[startIndiex:stopYIndex,:,:,:].cuda()
                
                else:

                    x_part = X[startIndiex:stopXIndex,:,:,:].cuda(non_blocking=True) #create a mini-batch of samples that fits on the GPU
                    
                    if len(Y) != 0:
                    
                        y_part = Y[startIndiex:stopYIndex,:,:].cuda()

                
                output = net(x_part).data #Run the samples though the network and get the predictions
                
                if key == 'test': # Only in testing mode
                    
                    if not('Scale' in params.architecture): # Architectures that do not zoom in the vessel in a second iteration. These architectures output a probability map without argmax that needs to be binarized
                    
                        output = torch.argmax(output, 1).cuda()
                    
                    if params.add3d == 0: # Full 2D+time architectures or full 2D architectures

                        net_results.append(output.cpu().numpy())

                        raw_files.append(x_part.cpu().numpy())

                        if len(Y) != 0:

                            ground_truths.append(y_part.cpu().numpy())
                            
                    else: # 2D+time models working with neighboring past and present frames
                        
                        net_results.append(output[:,:,:,params.add3d].cpu().numpy())

                        raw_files.append(x_part[:,:,:,:,params.add3d].cpu().numpy())

                        if len(Y) != 0:

                            ground_truths.append(y_part[:,:,:,params.add3d].cpu().numpy())

                    names.append(list(n)[0])
                    
                else: # Validation mode
                    
                    if not('Scale' in params.architecture):
                    
                        output = torch.argmax(output, 1).cuda() #returns the class with the highest probability and shrinks the tensor from (N, C(class probability), H, W) to (N, H, W)
                    
                if len(Y) != 0: # If ground-truths are available, compute metrics
                
                    for j in range(x_part.shape[0]):

                        for metric in metrics:

                            if metric == 'Dice' or metric == 'dice' or metric == 'DICE':

                                dice = Dice(y_part, output)

                                dices.append(dice.online())

                            elif metric == 'Precision' or metric == 'PRECISION' or metric == 'precision':

                                prec = Precision(y_part, output)

                                precisions.append(prec.online())


                            elif metric == 'Recall' or metric == 'recall' or metric == 'RECALL':

                                rec = Recall(y_part, output)

                                recalls.append(rec.online())


                            else:

                                print('Unspecified metric. Please provide an adequate metric (Dice/Precision/Recall)\n')

                                exit()
        
        
        if len(Y) != 0: # If ground-truths are available, compute the mean and STD of all the images evaluated and append them to the results
                            
            if ('Dice' in metrics) or ('dice' in metrics) or ('DICE' in metrics):

                mean_dice = np.mean(np.array(dices))

                std_dice = np.std(np.array(dices))

                name = 'dice'

                results.append([name, str(mean_dice) + ' ', str(std_dice) + ' ', iteration])



            if ('Precision' in metrics) or ('precision' in metrics) or ('PRECISION' in metrics):

                mean_prec = np.mean(np.array(precisions))

                std_prec = np.std(np.array(precisions))

                name = 'precision'

                results.append([name, str(mean_prec) + ' ', str(std_prec) + ' ', iteration])


            if ('Recall' in metrics) or ('recall' in metrics) or ('RECALL' in metrics):

                mean_rec = np.mean(np.array(recalls))

                std_rec = np.std(np.array(recalls))

                name = 'recall'

                results.append([name, str(mean_rec) + ' ', str(std_rec) + ' ', iteration])
    
    
    if key == 'val':
        
        params.test = False
        
        return results
    
    elif key == 'test':
        
        return results, raw_files, net_results, ground_truths, names
    
    else:
        
        print('Wrong key introduced. Please introduce "val" for model validation or "test" for model testing')
        
        
        
            
            


