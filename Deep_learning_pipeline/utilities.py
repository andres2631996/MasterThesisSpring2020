#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:31:28 2020

@author: andres
"""
import sys

import torch

import torch.nn.functional as F

import os

import params

import numpy as np

from torch.nn import BCELoss


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


def BCEloss(output, target):
    
    """
    Computes Binary Cross-Entropy loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Binary Cross-Entropy loss
    
    """
    
    return BCELoss(params.loss_weights)


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
    
    Ncl = output.shape[-1]
    
    w = np.zeros((Ncl,))
    
    target_array = target.numpy()
    
    for l in range(0,Ncl): w[l] = np.sum(target_array[:,:,:,:,l]==1,np.int8)
    
    w = 1/(w**2+0.00001)

    dice = dice_loss(output, target, weights = [w, w])
    
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


def tversky_loss_scalar(output, target):
    
    """
    Computes Tversky loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Dice loss
        
    """
    
    output = F.log_softmax(output, dim=1)[:,1]
    
    numerator = torch.sum(target * output)
    
    denominator = target * output + params.loss_beta * (1 - target) * output + (1 - params.loss_beta) * target * (1 - output)

    return 1 - (numerator + 1) / (torch.sum(denominator) + 1)


def focal_tversky_loss(output, target):
    
    """
    Computes Focal-Tversky loss.
    
    Params:
        
        - output: result from the network
        
        - target: ground truth result
        
    Returns:
        
        - Dice loss
        
    """
    
    tversky_loss = tversky_loss_scalar(output, target)
    
    return torch.pow(tversky_loss, params.loss_gamma)


def clear_GPU(net=None):
    
    sys.stdout.flush()
    
    if net !=None:
        
        for p in net.parameters():
            
            if p.grad is not None:
                
                del p.grad  # free some memory
                
    torch.cuda.empty_cache()
    
    
    

def load_checkpoint(model, optimizer, losslogger, filename = 'checkpoint.pth.tar'):
    
    """
    Load training checkpoint
    
    Params:
        
        - model: architecture from where checkpoint has been saved
        
        - optimizer: training optimizer state
        
        - losslogger: loss function state
        
        - filename: tar file name where model has been saved
    
    
    """
    
    
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    
    start_epoch = 0
    
    if os.path.isfile(filename):
        
        print("=> loading checkpoint '{}'".format(filename))
        
        checkpoint = torch.load(filename)
        
        start_epoch = checkpoint['iteration']
        
        model.load_state_dict(checkpoint['state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        loss = checkpoint['loss']
        
        print("=> loaded checkpoint '{}' (iteration {})"
                  .format(filename, checkpoint['iteration']))
    else:
        
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, loss
