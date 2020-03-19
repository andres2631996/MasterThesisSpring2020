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

import matplotlib.pyplot as plt

import itertools



def optimizerExtractor(net):
    
    """
    Return specified optimizer and if existing, learning rate scheduler.
    
    Params:
        
        - net: network where optimizer is applied
    
    
    """

    # Possible optimizers. Betas should not have to be changed
    
    found = 0
    
    if params.opt == 'Adam':
    
        optimizer = optim.Adam(net.parameters(), params.lr)
        
        found = 1
    
    elif params.opt == 'RMSprop':
        
        optimizer = optim.RMSprop(net.parameters(), params.lr)
        
        found = 1
    
    elif params.opt == 'SGD':
        
        optimizer = optim.SGD(net.parameters(), params.lr)
        
        found = 1
    
    else:
        
        print('\nWrong optimizer. Please define a valid optimizer (Adam/RMSprop/SGD)\n')
        
        exit()
    
    
    if found == 1:
        
        if params.lr_scheduling != False:
        
            if params.lr_scheduling == 'step':
            
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= params.step, gamma = params.lr_gamma)
        
            elif params.lr_scheduling == 'exponential':
            
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = params.lr_gamma)
        
            return optimizer, scheduler
        
        else:
            
            return optimizer



def lossSelection(output, Y):
    
    """
    Select the desired loss function among a set of possible options in the 
    Params file.
    
    Params:
        
        - output: result from network
        
        - Y: ground truth
    
    """
    
    # Select losses
    
    class_weights = torch.tensor(params.class_weights, dtype=torch.float32) # Weights to be used, if needed
    
    found = 0 # Flag to specify if the loss function has been found
            
    if params.loss_fun == 'dice':
        
        loss = utilities.dice_loss(output, Y.cuda(non_blocking=True), weights=class_weights.cuda(non_blocking=True))
        
        found = 1
    
    elif params.loss_fun == 'generalized_dice':
        
        loss = utilities.generalized_dice_loss(output, Y.cuda(non_blocking=True))
        
        found = 1
    
    elif params.loss_fun == 'bce':
        
        loss = utilities.BCEloss(output, Y.cuda(non_blocking=True))
        
        found = 1
    
    elif params.loss_fun == 'dice_bce':
        
        loss = utilities.DiceBCEloss(output, Y.cuda(non_blocking=True))
        
        found = 1
    
    elif params.loss_fun == 'exp_log':
        
        loss = utilities.exp_log_loss(output, Y.cuda(non_blocking=True))
        
        found = 1
    
    elif params.loss_fun == 'focal':
        
        loss = utilities.focal_loss(output, Y.cuda(non_blocking=True), weights=class_weights.cuda(non_blocking=True), gamma=params.loss_gamma)
        
        found = 1
    
    elif params.loss_fun == 'tversky':
        
        loss = utilities.tversky_loss_scalar(output, Y.cuda(non_blocking=True))
        
        found = 1
    
    elif params.loss_fun == 'focal_tversky':
        
        loss = utilities.focal_tversky_loss(output, Y.cuda(non_blocking=True))
        
        found = 1
    
    else:
        
        print('\nWrong loss. Please define a valid loss function (dice/generalized_dice/bce/dice_bce/exp_log/focal/tversky/focal_tversky)\n')

        exit()
    
    
    if found == 1:
        
        return loss
    

def checkpointLoading(net, optimizer, k):
    
    """
    Load checkpoints from previously trained models.
    
    Params:
        
        - net: network to be trained
        
        - optimizer: optimizer used to train the network
        
        - k: cross-validation fold from where to load model
    
    Returns:
        
        - net: loaded net
        
        - optimizer: loaded optimizer
    
    
    """


    print('\nTry to load previously trained models\n')
                    
    filename = 'trainedWith' + params.train_with + '_' + params.prep_step + 'fold_' + str(k) + '.tar' 
    
    files_saved = os.listdir(params.network_data_path)
    
    if filename in files_saved: # Model has been already run with the same parameters
        
        # Load previous model
        
        net, optimizer, start_epoch, loss, best_dice = utilities.load_checkpoint(net, optimizer, 
                                                                                   params.network_data_path,
                                                                                   filename)
        
        
        prev_dice = best_dice
    
    
    else:
        
        prev_dice = 0
        
        print('No previously trained model found\n')
        
        
    cont_load = 1
    
    return net, optimizer, cont_load, prev_dice



def train(net, loader_train, loader_val = None, k = 0):
    
    
    '''
    Trains a neural network and stores evaluation metrics
    
    :param net: network
    
    :param loader_train: training dataloader
    
    :param (optional) loader_val: validation dataloader, Default=None
    
    :param (optional) k: current iteration in K-fold crossvalidation, Default=0
    
    :returns: final_results_train: list with final result for training
    
    :returns: final_results_eval: list with final result for validation
    
    :returns: model: trained model state (if one wants to save it for later)
    
    :returns: optimizer: applied optimizer (if one wants to save it for later)
    
    :returns: optimizer_state: trained optimizer state (if one wants to save it for later)
    
    '''
    #Fetches parameters from params
    
    batch_size = params.batch_size
    
    batch_GPU = params.batch_GPU

    net.train(True) # Must be enabled for dropout to be active
    
    prev_dice = 0 # Reference Dice to save model
    
    # Extract optimizer and if existing, scheduler
    
    if params.lr_scheduling != False:
    
        optimizer, scheduler = optimizerExtractor(net) 
    
    else:
        
        optimizer = optimizerExtractor(net) 
    

    start_time = time.time()
    
    losses = [] # Stores loss from every iteration
    
    eval_metrics = [] # Stores dict of eval metrics from every [eval_frequency] step
    
    it = [] # Stores iteration indexes when network is evaluated
    
    it_loss = [] # Stores iteration indexes when loss is printed
    
    loss_print = [] # Average loss for printing
    
    loss_print_std = [] # Standard deviation loss for printing
    
    overall_results_val = [] # List with averaged validation metrics results per fold

    
    i = 0
    
    cont_load = 0
    
    #listenPaus = subprocess.Popen(["python3", "./listenPaus.py"])
    
    while i < params.I:
        
        #Loads a batch of samples into RAM
        
        for X, Y, name in loader_train:
            
            if i >= params.I:
                
                continue
                
            #Loops over the steps for this particular RAM-batch
            
            for j in range(int(X.shape[0]/batch_size)):

                if i >= params.I:
                    
                    continue

                optimizer.zero_grad()

                # Accumulates gradients for this minibatch 
                
                for m in range(math.ceil(batch_size/batch_GPU)):

                    i += batch_GPU 

                    if i >= params.I:
                        
                        continue
                    
                    #Sends a GPU-batch to the GPU
                    
                    startIndiex = j*batch_size + m*batch_GPU
                    
                    stopIndex = j*batch_size + min((m+1)*batch_GPU,batch_size)
                    
                    if params.three_D: # Training in 2D with one channel

                        
                        Xpart = X[startIndiex:stopIndex,:,:,:,:]
                
                        Ypart = Y[startIndiex:stopIndex,:,:]
                    
                    else:

                            
                        Xpart = X[startIndiex:stopIndex,:,:,:]
                    
                        Ypart = Y[startIndiex:stopIndex,:,:]
                        
                            
                    #Send sample(s) through the net
                    
                    output = net(Xpart.cuda(non_blocking=True))
                    
                    loss = lossSelection(output, Ypart) # Computed loss 
                    
                    losses.append(loss.item())

                    #Calculate gradients
                    
                    loss.backward()
                    
                #Perform update of weights
                
                optimizer.step()
                
                if params.lr_scheduling:
                    
                    scheduler.step()
                    
                # Loss printing
                
                if params.loss_frequency != -1 and loader_val != None and i % params.loss_frequency in range(batch_size) and i not in range(batch_size):

                    print("Training iteration {}: loss: {} +- {}\n".format(i, np.mean(losses[-params.loss_frequency:-1]), np.std(losses[-params.loss_frequency:-1])))
   
                    loss_print.append(np.mean(losses[-params.loss_frequency:-1]))
                    
                    loss_print_std.append(np.std(losses[-params.loss_frequency:-1]))
                    
                    it_loss.append(i)
                
                #Evaluates the network

                
                if cont_load == 0: # See if there is a previous model run and if so, load it
                    
                    net, optimizer, cont_load, prev_dice = checkpointLoading(net, optimizer, k)
                    
                    
                
                if (params.eval_frequency != -1 and loader_val != None and i % params.eval_frequency in range(batch_size) and i not in range(batch_size)) or (i == params.I - 1):
                    
                    #Evaluate a series of metrics at this point in training
                    
#                    plt.figure(figsize = (13,5))
#                    
#                    plt.subplot(131)
#                    
#                    plt.imshow(Xpart.cpu().detach().numpy()[0,0,:,:], cmap = 'gray')
#                    
#                    plt.subplot(132)
#                    
#                    plt.imshow(Ypart.cpu().numpy()[0,:,:], cmap = 'gray')
#                    
#                    plt.subplot(133)
#                    
#                    plt.imshow(out.cpu().detach().numpy()[0,:,:], cmap = 'gray')
                    
                    
                    #results_train = evaluate.evaluate(net, loader_train, i)
            
                    
                    
                    results_eval = evaluate.evaluate(net, loader_val, i)

                    eval_metrics.append(results_eval)
                    
                    #train_metrics.append(results_train)
                    
                    it.append(i)
                    
                    net.train(True)
                    
                    new_dice = float(results_eval[0][1][:-1])
                    
                    # Save the model if the validation score increases with respect to previous iterations
                        
                    if new_dice > prev_dice:
                        
                        state = {'iteration': i + 1, 'state_dict': net.state_dict(),
                                 'optimizer': optimizer.state_dict(), 'loss': loss, 
                                 'best_dice': new_dice}
                    
                        filename = 'trainedWith' + params.train_with + '_' + params.prep_step + 'fold_' + str(k) + '.tar' 
                        
                        torch.save(state, params.network_data_path + filename)
                        
                        print('Saved model\n')
                    
                        prev_dice = new_dice
                    
                    
                    for l in range(len(results_eval)):
                        
                        print("Validation {}: {} +- {}\n".
                              format(results_eval[l][0], results_eval[l][1], results_eval[l][2]))

                    print("Elapsed training time: {}\n".format(time.time() - start_time))
      


#            alive = listenPaus.poll()
#            
#            if alive == 0:
#                
#                del X
#                
#                del Y
#                
#                net.cpu()
#                
#                utilities.clear_GPU()
#                
#                print("Paused, press ENTER to resume or a number + ENTER to sleep that many seconds before resuming")
#                
#                while True:
#                    
#                    keystroke = input()
#                    
#                    if keystroke == "":
#                        
#                        break
#                    
#                    else:
#                        
#                        try:
#                            
#                            seconds = int(keystroke)
#                            
#                            print("Sleeping for " + keystroke + " seconds")
#                            
#                            time.sleep(seconds)
#                            
#                            break
#                        
#                        except ValueError:
#                            
#                            print("Press ENTER to resume or a number + ENTER to sleep that many seconds before resuming")
#                            
#                    print("Resuming")
                    
                #listenPaus = subprocess.Popen(["python3", "./listenPaus.py"])
                
                #net.cuda()

    # Remove one dimension from lists of results

    eval_metrics = list(itertools.chain.from_iterable(eval_metrics))

    
    print("Elapsed training time: {}".format(time.time() - start_time))
    
    # Saves training losses as a file
    

    with open(params.network_data_path + 'Training_losses_fold_' + str(k) + '_' + str(params.I) + 'trainedWith' + params.train_with + '_' + params.prep_step, 'w') as file:
        
        for i in range(len(losses)):
            
            if i != 0:
                
                file.write(',')
                
            file.write(str("%.20f" % losses[i]))
    
    
    # Plot losses and metrics and save their figures as .png
    
    # Loss
    
    plt.figure(figsize = (13,5))
    
    plt.errorbar(it_loss,loss_print, yerr= loss_print_std), plt.xlabel('Iterations'), plt.ylabel(params.loss_fun + ' loss')
    
    plt.title('Evolution of ' + params.loss_fun + ' loss function, fold ' + str(k))
    
    plt.savefig(params.network_data_path + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '_loss.png')
    
    # Metrics
    
    # Set metric lists to array
    
    eval_metrics_array = np.array(eval_metrics)


    
    # Save evaluation and training metrics to TXT files
    
    np.savetxt(params.network_data_path + 'ValidationMetrics_' + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '.txt', eval_metrics_array, fmt = '%s')
  
    # Get unique metrics names
    
    metrics_unique = np.unique(eval_metrics_array[:,0])
    
    # Apply a for loop for each metric type
    
    for metric_name in metrics_unique:
        
        metric_name = str(metric_name)
        
        ind_metric = np.where(eval_metrics_array[:,0] == metric_name)[0]
        
        mean_metric_eval = eval_metrics_array[ind_metric,1]
        
        std_metric_eval = eval_metrics_array[ind_metric,2]

        
        m_metric_eval = []
        
        s_metric_eval = []
        
        for i in range(len(mean_metric_eval)):
            
            m_metric_eval.append(float(mean_metric_eval[i][:-1]))
            
            s_metric_eval.append(float(std_metric_eval[i][:-1]))
        
        

        fig = plt.figure(figsize = (13,5))
        
        plt.errorbar(it, m_metric_eval, yerr =  s_metric_eval, color = 'r', label = 'Validation')
        
        plt.xlabel('Iterations'), plt.ylabel(metric_name)
        
        plt.title('Evolution of ' + metric_name + ' fold ' + str(k))
        
        plt.legend()
        
        fig.savefig(params.network_data_path + '/' + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '_' + metric_name + '.png')
        
        # Average result appending
        
        overall_results_val.append(m_metric_eval[-1])
        
        overall_results_val.append(s_metric_eval[-1])

    
    return np.mean(losses[-params.eval_frequency:-1]), np.std(losses[-params.eval_frequency:-1]), overall_results_val, net.state_dict(), optimizer, optimizer.state_dict()

