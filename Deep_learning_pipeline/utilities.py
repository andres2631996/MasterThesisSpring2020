#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:31:28 2020

@author: andres
"""
import sys

import torch

import torch.nn.functional as F

import torch.nn as nn

import os

import params

import numpy as np

from torch.nn import BCELoss

import matplotlib.pyplot as plt

from torch.autograd import Variable

import cc3d

from skimage import measure

from ripser import ripser

from persim import plot_diagrams

import scipy

import math

import warnings



def dice_loss(output, target, weights = params.loss_weights):
    
    """
    Computes normal Dice loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Dice loss
    
    """

    output_softmax = F.softmax(output, dim=1)

    output_flat = output_softmax[:,1].view(-1)
    
    target_flat = target.view(-1)
    
    intersection = 2*weights[1]*(torch.mul(output_flat, target_flat)).sum()
    
    denominator = torch.mul(output_flat, output_flat).sum() + torch.mul(target_flat, target_flat).sum()
    
    dice_loss = (intersection + 1)/(denominator + 1)
    
    return 1 - dice_loss


def BCEloss(output, target, balance = 1.1):
    
    """
    Computes Binary Cross-Entropy loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Binary Cross-Entropy loss
    
    """

    output_softmax = nn.Softmax(output)

    loss = BCELoss()(output_softmax, target) # initialize loss function

    return loss 




def DiceBCEloss(output, target):
    
    """
    Computes Dice + Binary Cross-Entropy loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Binary Cross-Entropy loss
    
    """
    
    bce_loss = BCEloss(output, target)
    
    diceLoss = dice_loss(output, target, weights = params.loss_weights)
    
    return bce_loss + diceLoss


def generalized_dice_loss(output, target):
    
    """
    Computes Generalized Dice loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Generalized Dice loss
    
    """

    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    
    Ncl = output.shape[1]
    
    w = np.zeros((Ncl,))
    
    target_array = target.numpy()
    
    weight = np.sum(target_array==1,np.int8)/np.prod(np.array(target.shape))
    
    w = [1-weight, weight]
    
    w = 1/(w**2+0.00001)

    dice = dice_loss(output, target, weights = w)
    
    return dice


def exp_log_loss(output, target):
    
    """
    Calculates exponential-logarithmic Dice loss
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Generalized Dice loss
    
    
    """
    
    weighted_dice = 0.8*torch.pow(-torch.log(torch.clamp(dice_loss(output,target), 1e-6)), params.loss_gamma_exp_log)
    
    weighted_bce = 0.2*torch.pow(-torch.log(torch.clamp(BCEloss(output,target), 1e-6)), params.loss_gamma_exp_log)
    
    return weighted_dice + weighted_bce


def connectedComponents(x):
    
    """
    Extract the number of connected components of a tensor.

    
    Params:
    
        - x: input tensor
        
    Returns:
    
        - out: number of connected components
    
    """ 

    # Transform tensor into Numpy array

    x_array = x.detach().cpu().numpy() # B,C,H,W (T)

    # Get labels from connected components for all elements in batch
    
    out = [] # Batch list with results

    if len(x_array.shape) == 3: # 2D arrays

        for i in range(x_array.shape[0]):

            labels = measure.label(x_array[i,:,:], background = 0)
            
            unique_labels = np.unique(labels.flatten())
            
            num_labels = len(unique_labels)
            
            out.append(num_labels)
            
            #probs = []

            #for label in unique_labels:

             #   ind_label = np.array(np.where(labels == label)) # Spatial coordinates of the same connected component

                # Extract the mean value of each connected component
                
              #  probs.append(np.mean(x_array[i,0,ind_label[0],ind_label[1]].flatten()))
                
            #prob_sorted = sorted(probs)
            
            #if len(probs) > 1:
                
             #   ind_max = probs.index(prob_sorted[-2]) # Index of connected component with the highest probability of being vessel

              #  label_max = unique_labels[ind_max] # Label number of the connected component with the highest probability

               # ind_max = np.array(np.where(labels == label_max)) # Spatial coordinates of connected component with the highest probability

               # out[i, ind_max[0], ind_max[1]] = 1
                
                

    elif len(x_array.shape) == 4: # 3D arrays

        for i in range(x_array.shape[0]):

            labels = cc3d.connected_components(x_array[i,:,:,:].astype(int)) # 26-connected
            
            unique_labels = np.unique(labels.flatten())
            
            num_labels = len(unique_labels)
            
            out.append(num_labels)
            
            #probs = []

            #for label in unique_labels:

             #   ind_label = np.array(np.where(labels == label)) # Spatial coordinates of the same connected component

                # Extract the mean value of each connected component
                
              #  probs.append(np.mean(x_array[i,1,ind_label[0],ind_label[1], ind_label[2]].flatten()))
                
            #if len(probs) > 1:
                
             #   prob_sorted = sorted(probs)

              #  ind_max = probs.index(prob_sorted[-2]) # Index of connected component with the highest probability of being vessel

               # label_max = unique_labels[ind_max] # Label number of the connected component with the highest probability

                #ind_max = np.array(np.where(labels == label_max)) # Spatial coordinates of connected component with the highest probability

                #out[i, ind_max[0], ind_max[1], ind_max[2]] = 1
            
    return out


def connectedComponentLoss(output, target, weight = 1):

    """
    Compute a loss based on the connected components of the network output and the target

    Params:

        - output: network output (torch.tensor)

        - target: mask (torch.tensor)

    """

    cc_output = connectedComponents(output)
    
    cc_target = connectedComponents(target)
    
    return weight*(np.sum(np.array(cc_output) - np.array(cc_target)))**2




def focal_loss(output, target, weights=None, size_average=True, gamma= 1/params.loss_gamma, eps = 10**-6):
    
    """
    Computes focal loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Focal loss
    
    """
    
    if gamma == 0:
        
        conf = torch.ones(output.size()).cuda()
        
    else:
        
        # Get confidences
        
        output_softmax =  F.softmax(output, dim=1)

        # Prevent sqrt(0) for gamma<1 which results in NaN  
        
        output_softmax_corr = output_softmax.sub(eps)

        conf = (torch.ones(output.size()).cuda() - output_softmax_corr) ** gamma
    
    # Get log predictions
    
    output = F.log_softmax(output, dim=1)

    # Weigh predictions by confidence
    
    output_conf = conf * output

    loss = F.nll_loss(output_conf, target, weights)
    
    return loss


def focal_cc_loss(output, target, weight = 0.1):
    
    """
    Combination of focal Dice overlap loss with Connected Components Loss
    
    Params:
    
        - output: result from the network
        
        - target: ground truth result
        
        - weight: weight for the Connected Components Loss
        
    Returns:
        
        - Combined Focal + Connected Component loss
    
    
    """
    
    focal = focal_loss(output, target)
    
    cc = connectedComponentLoss(output, target, 0.1)
    
    return focal + cc





def tversky_loss_scalar(output, target):
    
    """
    Computes Tversky loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Dice loss
        
    """
    
    output = F.log_softmax(output, dim=1)[:,0]
    
    numerator = torch.sum(target * output)
    
    denominator = target * output + params.loss_beta * (1 - target) * output + (1 - params.loss_beta) * target * (1 - output)

    return 1 - (numerator) / (torch.sum(denominator))


def focal_tversky_loss(output, target):
    
    """
    Computes Focal-Tversky loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Dice loss
        
    """
    
    tversky_loss = abs(tversky_loss_scalar(output, target))
    
    return torch.pow(tversky_loss, params.loss_gamma)


def clear_GPU(net=None):
    
    sys.stdout.flush()
    
    if net !=None:
        
        for p in net.parameters():
            
            if p.grad is not None:
                
                del p.grad  # free some memory
                
    torch.cuda.empty_cache()
    
    
    

def model_saving(model_state, optimizer_state, path, filename):

    """
    Save some model (and optimizer) after training.
    
    Params:
        
        - model: model to save
        
        - optimizer: optimizer
        
        - path: folder where to save model
        
        - filename: filename to save model (and optimizer)


    """ 
    
   
    state = {'state_dict': model_state, 'optimizer': optimizer_state}
    
    torch.save(state, path + filename)
    
    
    
    


def model_loading(model, optimizer, path, filename):
    
    """
    Load some model that has been saved AFTER TRAINING (NOT A CHECKPOINT)
    
    Usually used for inference
    
    Params:
        
        - model: architecture from where checkpoint has been saved
        
        - optimizer: training optimizer state
        
        - path: folder where model has been saved
        
        - filename: tar file name where model has been saved
    
    
    """
    
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    
    file = path + filename
    
    files = os.listdir(path)

    if os.path.exists(path):
    
        if filename in files:
            
            print("=> loading model '{}'".format(filename))
            
            checkpoint = torch.load(file)
            
            model.load_state_dict(checkpoint['state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        else:
            
            print('Non-existing file in path given. Please provide a valid path')
    
    else:
        
        print('Non-existing path. Please provide a valid path')
  

    return model, optimizer
    
        
        

def load_checkpoint(model, optimizer, path, filename = 'checkpoint.pth.tar'):
    
    """
    Load training checkpoint or model for inference
    
    Params:
        
        - model: architecture from where checkpoint has been saved
        
        - optimizer: training optimizer state
        
        - path: folder where model has been saved
        
        - filename: tar file name where model has been saved
    
    
    """
    
    
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    
    file = path + filename
    
    files = os.listdir(path)
    
    if os.path.exists(path):
    
        if filename in files:
            
            print("=> loading checkpoint '{}'\n".format(filename))
            
            checkpoint = torch.load(file)
            
            start_epoch = checkpoint['iteration']
            
            model.load_state_dict(checkpoint['state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            best_dice = checkpoint['best_dice']
            
            loss = checkpoint['loss']
            
            print("=> loaded checkpoint '{}' (iteration {}, dice {})\n"
                      .format(filename, checkpoint['iteration'],
                              checkpoint['best_dice']))
            
            return model, optimizer, start_epoch, loss, best_dice
        
        
        else:
            
            print('Non-existing file in path given. Please provide a valid path')
    
    else:
        
        print('Non-existing path. Please provide a valid path')
  




def print_num_params(model, display_all_modules=False):
    
    
    """
    Print total number of model parameters.
    
    Params:
        
        - model: model from where we want to see the number of parameters
    
    Returns:
        
        - printed number of parameters
    
    """
    
    total_num_params = 0
    
    for n, p in model.named_parameters():
        
        num_params = 1
        
        for s in p.shape:
            
            num_params *= s
            
        if display_all_modules: print("{}: {}".format(n, num_params))
        
        total_num_params += num_params
        
    print("-" * 50)
    print("Total number of parameters: {:.2e}".format(total_num_params))
    
    
    
    

def loadMetricsResults(path, file):
    
    """
    Load training or validation metrics results from some previous run.
    
    Params:
        
        - path: folder where txt with results is saved
        
        - file: txt file to analyze
    
    
    Returns:
        
        - names: array with metrics' names
        
        - mean_values: array with metrics' mean values
        
        - std_values: array with metrics' standard deviation values
        
        - it_values: array with iteration numbers
        
        - additionally, Matplotlib figures with plotted results are also saved 
        as PNG in the same folder as the TXT file
    
    
    
    """
    
    
    if file[-3:] == 'txt':
        
        array = np.loadtxt(path + file, dtype = str)
            
        # Transform values into float
        
        mean_values = array[:,1].astype(np.float)
        
        std_values = array[:,2].astype(np.float)
        
        it_values = array[:,3].astype(np.float)
        
        # Get unique metrics names
        
        metrics_unique = np.unique(array[:,0])
        
        # Apply a for loop for each metric type
        
        for metric_name in metrics_unique:
            
            metric_name = str(metric_name)
            
            ind_metric = np.where(array[:,0] == metric_name)[0]
    
            mean_metric = mean_values[ind_metric]
            
            std_metric = std_values[ind_metric]
            
            it_metric = it_values[ind_metric]
    
            fig = plt.figure(figsize = (13,5))
            
            if 'Training' in file:
            
                plt.errorbar(it_metric, mean_metric, yerr =  std_metric, color ='b', label = 'Training')
            
            elif 'Validation' in file:
                
                plt.errorbar(it_metric, mean_metric, yerr =  std_metric, color = 'r', label = 'Validation')
            
            else:
                
                print('Wrong .txt file introduced. Please introduce a valid .txt file for loading')
            
            plt.xlabel('Iterations'), plt.ylabel(metric_name)
            
            plt.title('Evolution of ' + metric_name)
            
            plt.legend()
            
            final_filename = path + file.replace('txt','png')
            
            fig.savefig(final_filename)
            
            return array[:,0], mean_values, std_values, it_values
            
    
    else:
        
        print('Introduced file is not TXT. Please introduce a TXT file')
        

        
        
def connectedComponentsPostProcessing(x):
    
    """
    Extract the number of connected components of a tensor.

    
    Params:
    
        - x: input tensor
        
    Returns:
    
        - out: number of connected components
    
    """ 

    # Transform tensor into Numpy array

    x_array = x.cpu().numpy() # B,C,H,W (T)
    
    binary_array = torch.argmax(x, 1).cpu().numpy() # Inference output

    # Get labels from connected components for all elements in batch
    
    out = torch.zeros(binary_array.shape)

    if len(x_array.shape) == 4: # 2D arrays

        for i in range(x_array.shape[0]):

            labels = measure.label(x_array[i,:,:], background = 0)
            
            unique_labels = np.unique(labels.flatten())
            
            num_labels = len(unique_labels)
            
            probs = []

            for label in unique_labels:

                ind_label = np.array(np.where(labels == label)) # Spatial coordinates of the same connected component

                # Extract the mean value of each connected component
                
                probs.append(np.mean(x_array[i,0,ind_label[0],ind_label[1]].flatten()))
                
            prob_sorted = sorted(probs)
            
            if len(probs) > 1:
                
                ind_max = probs.index(prob_sorted[-2]) # Index of connected component with the highest probability of being vessel

                label_max = unique_labels[ind_max] # Label number of the connected component with the highest probability

                ind_max = np.array(np.where(labels == label_max)) # Spatial coordinates of connected component with the highest probability

                out[i, ind_max[0], ind_max[1]] = 1
                
                

    elif len(x_array.shape) == 5: # 3D arrays

        for i in range(x_array.shape[0]):

            labels = cc3d.connected_components(binary_array[i,:,:,:].astype(int)) # 26-connected
            
            median = np.median(labels.flatten()) # Label with background, to be excluded
            
            unique_labels = np.unique(labels.flatten())
            
            unique_labels = np.delete(unique_labels, np.where(unique_labels == median)) 
            
            num_labels = len(unique_labels)
            
            probs = []

            for label in unique_labels:

                ind_label = np.array(np.where(labels == label)) # Spatial coordinates of the same connected component

                # Extract the mean value of each connected component
                
                probs.append(np.mean(x_array[i,1,ind_label[0],ind_label[1], ind_label[2]].flatten()))

            print(probs)
            
            plt.figure(figsize = (13,5))
            
            plt.imshow(np.mean(labels, axis = -1)), plt.colorbar()
            
            if len(probs) > 0:
                
                prob_sorted = sorted(probs)

                ind_max = probs.index(max(probs)) # Index of connected component with the highest probability of being vessel

                label_max = unique_labels[ind_max] # Label number of the connected component with the highest probability

                ind_max = np.array(np.where(labels == label_max)) # Spatial coordinates of connected component with the highest probability

                out[i, ind_max[0], ind_max[1], ind_max[2]] = 1
            
    return out


#def persistent_homology(



def distanceTransform(tensor, key):
    
    """
    Compute distance transform of tensor, normalizing it from -1 to +1
    
    Params:
    
        - tensor: input tensor
        
        - key: variable specifying if the computation is inside the network (loss/net)
        
    Output:
    
        - trans: distance transform output
    
    """
    
    # Disallow gradients from this section
    
    #with torch.no_grad():
    
    if key == 'net' or key == 'NET' or key == 'Net':
        
        numpy_tensor = tensor.detach().cpu().numpy()
        
    elif key == 'loss' or key == 'LOSS' or key == 'Loss':
        
        numpy_tensor = tensor.cpu().numpy()

    batch_len = numpy_tensor.shape[0]

    if key == 'net' or key == 'NET' or key == 'Net':

        classes = numpy_tensor.shape[1]

    elif key == 'loss' or key == 'LOSS' or key == 'Loss':

        classes = 1

    trans_numpy = np.zeros(numpy_tensor.shape)

    trans_numpy_inv = np.zeros(numpy_tensor.shape)

    for batch_ind in range(batch_len):

        for cl in range(classes):

            if key == 'net' or key == 'NET' or key == 'Net':

                if len(numpy_tensor.shape) == 5:

                    numpy_tensor[batch_ind, cl, :, :, :] = (numpy_tensor[batch_ind, cl, :, :, :] - np.amin(numpy_tensor[batch_ind, cl, :, :, :]))/(np.amax(numpy_tensor[batch_ind, cl, :, :, :]) - np.amin(numpy_tensor[batch_ind, cl, :, :, :]))

                    trans_numpy[batch_ind, cl, :, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor[batch_ind, cl, :, :, :])


                elif len(numpy_tensor.shape) == 4:

                    numpy_tensor[batch_ind, cl, :, :] = (numpy_tensor[batch_ind, cl, :, :] - np.amin(numpy_tensor[batch_ind, cl, :, :]))/(np.amax(numpy_tensor[batch_ind, cl, :, :]) - np.amin(numpy_tensor[batch_ind, cl, :, :]))


                    trans_numpy[batch_ind, cl, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor[batch_ind, cl, :, :])

                trans_numpy_final = np.copy(trans_numpy)  

            elif key == 'loss' or key == 'LOSS' or key == 'Loss':

                if len(numpy_tensor.shape) == 4:

                    numpy_tensor[batch_ind, :, :, :] = (numpy_tensor[batch_ind, :, :, :] - np.amin(numpy_tensor[batch_ind, :, :, :]))/(np.amax(numpy_tensor[batch_ind, :, :, :]) - np.amin(numpy_tensor[batch_ind, :, :, :]))

                    numpy_tensor_inv = np.abs(1 - numpy_tensor) # Inverted tensor

                    trans_numpy[batch_ind, :, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor[batch_ind, :, :, :])

                    trans_numpy_inv[batch_ind, :, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor_inv[batch_ind, :, :, :])

                elif len(numpy_tensor.shape) == 3:

                    numpy_tensor[batch_ind, :, :] = (numpy_tensor[batch_ind, :, :] - np.amin(numpy_tensor[batch_ind, :, :]))/(np.amax(numpy_tensor[batch_ind, :, :]) - np.amin(numpy_tensor[batch_ind, :, :]))

                    numpy_tensor_inv = np.abs(1 - numpy_tensor) # Inverted tensor

                    trans_numpy[batch_ind, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor[batch_ind, :, :])

                    trans_numpy_inv[batch_ind, :, :] = scipy.ndimage.morphology.distance_transform_edt(numpy_tensor_inv[batch_ind, :, :])

                trans_numpy_final = trans_numpy - trans_numpy_inv


    # Normalization from -1 to +1
    
    if np.sum(trans_numpy_final.flatten()) > 0:
        
        warnings.filterwarnings("ignore")

        trans_numpy_final = (trans_numpy_final - np.amin(trans_numpy_final))/(np.amax(trans_numpy_final) - np.amin(trans_numpy_final))

        trans_numpy_final = 2*trans_numpy_final - 1

        trans = torch.tensor(trans_numpy_final, requires_grad = True, device = 'cuda:0')
        
    else:
        
        # Avoid empty distance transforms
        
        trans = torch.ones(numpy_tensor.shape, requires_grad = True, device = 'cuda:0')
        

 
        
    return trans
                        

    
def distance_loss(output, target):
    
    """
    Computes Distance loss to reduce the impact of false positives and refine vessel boundaries.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Distance loss
        
    """
    
    output = torch.argmax(output, 1)
    
    output_dist = distanceTransform(output, 'loss')
    
    #output_dist = torch.tensor(output_dist, requires_grad = True) # Allow for backprop
    
    target_dist = distanceTransform(target, 'loss')
    
    l1_loss = nn.L1Loss()
    
    return l1_loss(output_dist, target_dist)



def focal_distance_loss(output, target, iteration, total):
    
    
    """
    Computes Focal + Distance loss to improve result overlap while reducing the impact of false positives and refine vessel boundaries. The distance loss will be weighted with a scheduling process
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
        - iteration: iteration number
        
        - total: total number of training iterations
        
    Returns:
        
        - Distance loss
        
    """
    
    focal = focal_loss(output, target)
    
    distance = distance_loss(output, target)
    
    
    if iteration < total//2:
        
        # Before reaching half of the iterations, increasing weight
        
        distance_weight = 2*iteration/total
        
    else:
        
        # After half of iterations, weight of 1. Same importance as focal loss
        
        distance_weight = 1

        
   
    return focal + distance_weight*distance