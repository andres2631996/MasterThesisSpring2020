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


class Dice():
    
    """
    Compute Dice Coefficient as metrics.
    
    Params:
        
        - ground_truth: mask
        
        - prediction: network output
    
    
    Returns:
        
        - dice coefficient value
    
    """
    
    def __init__(self, ground_truth, prediction):
        
        self.ground_truth = ground_truth
        
        self.prediction = prediction
        
    def online(self):
        
        numerator = 0
        
        denominator = 0
        
        numerator += float(torch.sum(torch.mul(self.ground_truth, self.prediction)).cpu())
        
        denominator += float(torch.add(torch.sum(self.ground_truth), torch.sum(self.prediction)).cpu())
        
        return 2*numerator/(denominator + 0.000000001)
    
    
    


class Precision():
    
    """
    Compute precision value as metrics
    
    Params:
        
        - ground_truth: mask
        
        - prediction: network output
        
    Returns:
    
        - precision value
    
    """
    
    def __init__(self, ground_truth, prediction):
        
        self.ground_truth = ground_truth
        
        self.prediction = prediction
    
    
    def online(self):
        
        numerator = 0
        
        denominator = 0
        
        numerator += float(torch.sum(torch.mul(self.ground_truth, self.prediction)).cpu())
        
        denominator += float(torch.sum(self.prediction).cpu())
        
        return numerator/(denominator + 0.000000001)
    
    


class Recall():
    
    """
    Compute recall value as metrics
    
    Params:
        
        - ground_truth: mask
        
        - prediction: network output
        
    Returns:
    
        - recall value
    
    """
    
    def __init__(self, ground_truth, prediction):
        
        self.ground_truth = ground_truth
        
        self.prediction = prediction
    
    
    def online(self):
        
        numerator = 0
        
        denominator = 0
        
        numerator += float(torch.sum(torch.mul(self.ground_truth, self.prediction)).cpu())
        
        denominator += float(torch.sum(self.ground_truth).cpu())
        
        return numerator/(denominator + 0.000000001)
        


class TrueVolume():
    
    def __init__(self):
        
        self.voxels = 0
        
    def online(self, ground_truth, prediction):
        
        self.voxels += int(torch.sum(ground_truth).cpu())
        
    def final(self):
        
        return self.voxels
    
    

class PredictedVolume():
    
    def __init__(self):
        
        self.voxels = 0
        
    def online(self, ground_truth, prediction):
        
        self.voxels += int(torch.sum(prediction).cpu())
        
        
    def final(self):
        
        return self.voxels





class MIP():
    
    def __init__(self, shapes):
        
        self.input = torch.zeros(shapes).cuda()#.long().cuda()
        
        self.seg = torch.zeros(shapes).long().cuda()
        
    def online(self, ground_truth, prediction, raw):
        
        #input_data = input_data.long()
   
        if 'both' in params.train_with:
    
            self.input = torch.max(self.input, raw[:,:,:,0])
        
        else:
            
            self.input = torch.max(self.input, raw[:,:,:])
            

        correct = 3 * torch.mul(ground_truth, prediction)
        
        under = 2 * ground_truth
        
        over = prediction
        
        temp = torch.max(under, over)
        
        temp = torch.max(temp, correct)
        
        self.seg = torch.max(temp, self.seg)
        
    def final(self):
        
        self.input = self.input.cpu().numpy()
        
        self.seg = self.seg.cpu().numpy()
        
        return self
    
    
  
    

class Segmentations():
    
    def __init__(self):
        
        self.input = []
        
        self.seg = []

    def online(self, ground_truth, prediction, raw):
        
        #input_data = input_data.long()
        
        if 'both' in params.train_with:
            
            if params.three_D:
            
                for k in range(raw.shape[3]):
            
                    self.input.append(raw[:, :, :, k, 0].cpu().numpy())
            
            else:
                
                for k in range(raw.shape[2]):
            
                    self.input.append(raw[:, :, k, 0].cpu().numpy())
        
        else:
            
            if params.three_D:
                
                for k in range(raw.shape[3]):
            
                    self.input.append(raw[:, :, :, k].cpu().numpy())
                
            else:
                
                for k in range(raw.shape[2]):
            
                    self.input.append(raw[:, :, k].cpu().numpy())
                
        
        correct = 3 * torch.mul(ground_truth, prediction)
        
        under = 2 * ground_truth
        
        over = prediction
        
        temp = torch.max(under, over)
        
        temp = torch.max(temp, correct)
        
        self.seg.append(temp.cpu().numpy())
        
    def final(self):
        
        return self



def evaluate(net, loader, iteration):
    
    """
    Evaluations done during cross-validation
    
    :param net:
        
    :param loader:
        
    :param iteration:

    :return:
    """
    
    
    with torch.no_grad():
        
        net.eval()  # This is necessary for the dropout layers to work

        batch_gpu_max = params.batch_GPU_max_inference
        
        ram_batch_size = params.RAM_batch_size
        
        metrics = params.metrics
        
        dices = [] # List with evaluated Dice values
        
        precisions = [] # List with evaluated precisions
        
        recalls = [] # List with evaluated recalls
        
        results = [] # List with overall results: METRIC NAME, MEAN AND STD!!

        for X, Y, _ in loader:
            
            # Extracts the data from dataloader and puts it into the network
            
            for i in range(min(math.ceil(ram_batch_size/batch_gpu_max), math.ceil(X.shape[0]/batch_gpu_max))):
                
                startIndiex = i*batch_gpu_max
                
                stopXIndex = min((i+1)*batch_gpu_max, X.shape[0])
                
                stopYIndex = min((i+1)*batch_gpu_max, Y.shape[0])
                
                if params.three_D: # Training in 2D with one channel
    
                    x_part = X[startIndiex:stopXIndex,:,:,:,:].cuda(non_blocking=True) #create a mini-batchof samples that fits on the GPU

                    y_part = Y[startIndiex:stopYIndex,:,:,:].cuda()
                
                else:

                    x_part = X[startIndiex:stopXIndex,:,:,:].cuda(non_blocking=True) #create a mini-batchof samples that fits on the GPU
                    
                    y_part = Y[startIndiex:stopYIndex,:,:].cuda()
                
                output = net(x_part).data #Run the samples though the network and get the predictions 
                
                output = torch.argmax(output, 1) #returns the class with the highest probability and shrinks the tensor from (N, C(class probability), H, W) to (N, H, W)
                
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
        
        
    return results
        
        
        
            
            


