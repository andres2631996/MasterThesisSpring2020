#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:55:58 2020

@author: andres
"""

import os

import sys

import subprocess

import numpy as np

import math

import time

import datetime

import torch

import torch.optim as optim

import torch.nn.functional as F

import evaluate

import utilities

import params

import gc

import matplotlib.pyplot as plt


def train(net, loader_train, loader_val = None, k = 0, eval_frequency = params.eval_frequency, I = params.I, prev_dice = 0):
    
    
    '''
    Trains a neural network and stores evaluation metrics
    
    :param net: network
    
    :param loader_train: training dataloader
    
    :param (optional) loader_val: validation dataloader, Default=None
    
    :param (optional) k: current iteration in K-fold crossvalidation, Default=0 
    
    :returns: list(iteration) of dicts(metric) of dicts(patient) containing evaluation results
    
    '''
    #Fetches parameters from params
    
    batch_size = params.batch_size
    
    batch_GPU = params.batch_GPU
    
    class_weights = torch.tensor(params.class_weights, dtype=torch.float32)
    
    class_count = len(params.class_weights)

    net.train(True) # Must be enabled for dropout to be active
    

    # Possible optimizers. Betas should not have to be changed
    
    if params.opt == 'Adam':
    
        optimizer = optim.Adam(net.parameters(), params.lr)
    
    elif params.opt == 'RMSprop':
        
        optimizer = optim.RMSprop(net.parameters(), params.lr)
    
    elif params.opt == 'SGD':
        
        optimizer = optim.SGD(net.parameters(), params.lr)
    
    else:
        
        print('\nWrong optimizer. Please define a valid optimizer (Adam/RMSprop/SGD)\n')
        
        
    # Possibility of learning rate scheduling
    
        
    if params.lr_scheduling == 'step':
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= params.step, gamma = params.lr_gamma)
    
    elif params.lr_scheduling == 'exponential':
        
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = params.lr_gamma)
        
        
        

    start_time = time.time()
    
    losses = [] # Stores loss from every iteration
    
    eval_metrics = [] # Stores dict of eval metrics from every [eval_frequency] step
    
    train_metrics = [] # Stores dict of train metrics from every [eval_frequency] step
    
    i = 0
    
    listenPaus = subprocess.Popen(["python3", "./listenPaus.py"])
    
    while i < I:
        
        #Loads a batch of samples into RAM
        
        for X, Y, _ in loader_train:
            
            if i >= I:
                
                continue
                
            #Loops over the steps for this particular RAM-batch
            
            for j in range(int(X.shape[0]/batch_size)):

                if i >= I:
                    
                    continue

                optimizer.zero_grad()

                #Accumulates gradients for this minibatch 
                
                for m in range(math.ceil(batch_size/batch_GPU)):

                    i += batch_GPU 

                    if i >= I:
                        
                        continue
                    
                    #Sends a GPU-batch to the GPU
                    
                    startIndiex = j*batch_size + m*batch_GPU
                    
                    stopIndex = j*batch_size + min((m+1)*batch_GPU,batch_size)
                    
                    if params.three_D: # Training in 2D with one channel
                        
                        if 'both' in params.train_with:
                    
                            Xpart = X[startIndiex:stopIndex,:,:,:,:]

                        
                        else:
                            
                            Xpart = X[startIndiex:stopIndex,:,:,:]
                    
                        Ypart = Y[startIndiex:stopIndex,:,:,:]
                    
                    else:
                        
                        if 'both' in params.train_with:
                            
                            Xpart = X[startIndiex:stopIndex,:,:,:]
                        
                        else:
                            
                            Xpart = X[startIndiex:stopIndex,:,:]
                        
                        Ypart = Y[startIndiex:stopIndex,:,:]
                        
                            
                    #Send sample(s) through the net
                    
                    output = net(Xpart.cuda(non_blocking=True))
                    
                    # Select losses
                    
                    if params.loss_fun == 'dice':
                        
                        loss = utilities.dice_loss(output, Ypart.cuda(non_blocking=True), weights=class_weights.cuda(non_blocking=True))
                    
                    elif params.loss_fun == 'generalized_dice':
                        
                        loss = utilities.generalized_dice_loss(output, Ypart.cuda(non_blocking=True))
                    
                    elif params.loss_fun == 'bce':
                        
                        loss = utilities.BCEloss(output, Ypart.cuda(non_blocking=True))
                    
                    elif params.loss_fun == 'dice_bce':
                        
                        loss = utilities.DiceBCEloss(output, Ypart.cuda(non_blocking=True))
                    
                    elif params.loss_fun == 'exp_log':
                        
                        loss = utilities.exp_log_loss(output, Ypart.cuda(non_blocking=True))
                    
                    elif params.loss_fun == 'focal':
                        
                        loss = utilities.focal_loss(output, Ypart.cuda(non_blocking=True), weights=class_weights.cuda(non_blocking=True), gamma=params.loss_gamma)
                    
                    elif params.loss_fun == 'tversky':
                        
                        loss = utilities.tversky_loss_scalar(output, Ypart.cuda(non_blocking=True))
                    
                    elif params.loss_fun == 'focal_tversky':
                        
                        loss = utilities.focal_tversky_loss(output, Ypart.cuda(non_blocking=True))
                    
                    else:
                        
                        print('\nWrong loss. Please define a valid loss function (dice/generalized_dice/bce/dice_bce/exp_log/focal/tversky/focal_tversky)\n')
                        
                    losses.append(loss.item())

                    #Calculate gradients
                    
                    loss.backward()
                    
                #Perform update of weights
                
                optimizer.step()
                
                if params.lr_scheduling:
                    
                    scheduler.step()
                
                #Evaluates the network
                    
#                    if loader_val == None and i%eval_frequency in range(batch_size):
#                        
#                        print("Training iteration {}: loss: {}".format(i, losses[-1]))
#                        
#                        print("Elapsed training time: {}".format(time.time() - start_time))

                
                if eval_frequency != -1 and loader_val != None and i % eval_frequency in range(batch_size) and i not in range(batch_size):

                    print("Training iteration {}: loss: {}".format(i, losses[-1]))
                    
                    print("Elapsed training time: {}".format(time.time() - start_time))

                    #Evaluate a dice score at this point in training
                
                    eval_metrics.append(evaluate.evaluateCrossvalidation(net, loader_val, class_count, params.metrics))
                    
                    train_metrics.append(evaluate.evaluateCrossvalidation(net, loader_train, class_count, params.metrics))
                    
                    net.train(True)
                    
                    # Save the model if the validation score increases with respect to previous iterations
                    
                    if evaluate.evaluateCrossvalidation(net, loader_val, class_count, params.metrics) > prev_dice:
                    
                        state = {'iteration': i + 1, 'state_dict': net.state_dict(),
                                 'optimizer': optimizer.state_dict(), 'loss': loss}
                    
                        filename = params.network_data_path + 'trainedWith' + params.train_with + '_' + params.prep_step + '.tar' 
                    
                        torch.save(state, filename)

                
                prev_dice = evaluate.evaluateCrossvalidation(net, loader_val, class_count, params.metrics) # Look for some index!!!


            alive = listenPaus.poll()
            
            if alive == 0:
                
                del X
                
                del Y
                
                net.cpu()
                
                utilities.clear_GPU()
                
                print("Paused, press ENTER to resume or a number + ENTER to sleep that many seconds before resuming")
                
                while True:
                    
                    keystroke = input()
                    
                    if keystroke == "":
                        
                        break
                    
                    else:
                        
                        try:
                            
                            seconds = int(keystroke)
                            
                            print("Sleeping for " + keystroke + " seconds")
                            
                            time.sleep(seconds)
                            
                            break
                        
                        except ValueError:
                            
                            print("Press ENTER to resume or a number + ENTER to sleep that many seconds before resuming")
                            
                    print("Resuming")
                    
                listenPaus = subprocess.Popen(["python3", "./listenPaus.py"])
                
                net.cuda()


    #Final evaluation for creating MIPs etc
    
    if loader_val != None:
        
        evaluate.evaluate_final(net, loader_val, class_count, params.final_metrics, params.data_path)

    listenPaus.kill()

    end_time = time.time()
    
    print("Elapsed training time: {}".format(end_time - start_time))
    
    # Saves training losses as a file
    

    with open(params.data_path + 'Training_losses_fold_' + str(k) + '_' + str(I) + 'trainedWith' + params.train_with + '_' + params.prep_step, 'w') as file:
        
        for i in range(len(losses)):
            
            if i != 0:
                
                file.write(',')
                
            file.write(str("%.20f" % losses[i]))
    
    
    # Plot losses and metrics and save their figures as .png
    
    # Loss
    
    plt.figure(figsize = (13,5))
    
    plt.plot(range(1,len(loss)),loss), plt.xlabel('Iterations'), plt.ylabel('Loss values')
    
    plt.title('Evolution of loss function')
    
    plt.savefig(params.network_data_path + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '_loss.png')
    
    # Metrics
    
    dice_train = train_metrics[0]
    
    precision_train = train_metrics[1]
    
    recall_train = train_metrics[2]
    
    dice_eval = eval_metrics[0]
    
    precision_eval = eval_metrics[1]
    
    recall_eval = eval_metrics[2]
    
    
    plt.figure(figsize = (13,5))
    
    plt.plot(range(1,len(dice_train)*params.eval_frequency,eval_frequency),dice_train, 'b', label = 'Training')
    
    plt.plot(range(1,len(dice_eval)*params.eval_frequency,eval_frequency),dice_eval, 'r', label = 'Validation')
    
    plt.xlabel('Iterations'), plt.ylabel('Dice coefficient')
    
    plt.title('Evolution of Dice coefficient')
    
    plt.savefig(params.network_data_path + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '_dice.png')
    
    
    
    plt.figure(figsize = (13,5))
    
    plt.plot(range(1,len(precision_train)*params.eval_frequency,eval_frequency),precision_train, 'b', label = 'Training')
    
    plt.plot(range(1,len(precision_eval)*params.eval_frequency,eval_frequency),precision_eval, 'r', label = 'Validation')
    
    plt.xlabel('Iterations'), plt.ylabel('Precision')
    
    plt.title('Evolution of precision')
    
    plt.savefig(params.network_data_path + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '_precision.png')
    
    
    
    plt.figure(figsize = (13,5))
    
    plt.plot(range(1,len(recall_train)*params.eval_frequency,eval_frequency),recall_train, 'b', label = 'Training')
    
    plt.plot(range(1,len(recall_eval)*params.eval_frequency,eval_frequency),recall_eval, 'r', label = 'Validation')
    
    plt.xlabel('Iterations'), plt.ylabel('Recall')
    
    plt.title('Evolution of recall')
    
    plt.savefig(params.network_data_path + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '_recall.png')
    
    
    return eval_metrics

