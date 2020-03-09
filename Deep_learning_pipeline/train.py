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



def train(net, loader_train, loader_val = None, k = 0, eval_frequency = params.eval_frequency, I = params.I):
    
    
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
    
    class_weights = torch.tensor(params.class_weights, dtype=torch.float32)

    net.train(True) # Must be enabled for dropout to be active
    
    prev_dice = 0
    
    
    cont_load = 0 # Flag to load previous models
    
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
    
    it = [] # Stores iteration indexes when network is evaluated

    
    i = 0
    
    #listenPaus = subprocess.Popen(["python3", "./listenPaus.py"])
    
    while i < I:
        
        #Loads a batch of samples into RAM
        
        for X, Y, name in loader_train:
            
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

                        
                        Xpart = X[startIndiex:stopIndex,:,:,:,:]
                
                        Ypart = Y[startIndiex:stopIndex,:,:]
                    
                    else:

                            
                        Xpart = X[startIndiex:stopIndex,:,:,:]
                    
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
                    
                    out = torch.argmax(output, 1)
                    
                    
                    results_eval = evaluate.evaluate(net, loader_val, i)
                    
                    results_train = evaluate.evaluate(net, loader_train, i)
                
                    eval_metrics.append(results_eval)
                    
                    train_metrics.append(results_train)
                    
                    it.append(i)
                    
                    net.train(True)
                    
                    new_dice = float(results_eval[0][1][:-1])
                    
                    # Save the model if the validation score increases with respect to previous iterations
                    
                    if new_dice > prev_dice:
                    
                        state = {'iteration': i + 1, 'state_dict': net.state_dict(),
                                 'optimizer': optimizer.state_dict(), 'loss': loss, 
                                 'best_dice': new_dice}
                    
                        filename = 'trainedWith' + params.train_with + '_' + params.prep_step + 'fold_' + str(k) + '.tar' 
                        
                        files_saved = os.listdir(params.network_data_path)
                        
                        if filename in files_saved and cont_load != 1: # Model has been already run with the same parameters
                            
                            # Load previous model
                            
                            net, optimizer, start_epoch, loss, best_dice = utilities.load_checkpoint(net, optimizer, 
                                                                                                       params.network_data_path,
                                                                                                       filename)

                                
                            prev_dice = best_dice
                            
                            cont_load = 1
                        
                        else:
                            
                            print('Saved model')
                            
                            torch.save(state, params.network_data_path + filename)
                        
                            prev_dice = new_dice
                    
                    print("Training iteration {}: loss: {}\n".format(i, losses[-1]))
                    
                    for l in range(len(results_eval)):
                        
                        print("Training {}: {} +- {} / Validation {}: {} +- {}\n".
                              format(results_train[l][0], results_train[l][1], results_train[l][2], 
                                     results_eval[l][0], results_eval[l][1], results_eval[l][2]))

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
    
    train_metrics = list(itertools.chain.from_iterable(train_metrics))

    #Final evaluation for creating MIPs etc
    
    if loader_val != None:
        
        final_results_eval = evaluate.evaluate(net, loader_val, I)
    
    final_results_train = evaluate.evaluate(net, loader_train, I)

    #listenPaus.kill()

    end_time = time.time()
    
    print('Final loss: {}\n'.format(losses[-1]))
    
    for i in range(len(final_results_eval)):
                        
        print("Final Training {}: {} +- {} / Final Validation {}: {} +- {}\n".
              format(final_results_train[i][0], final_results_train[i][1], 
                     final_results_train[i][2], final_results_eval[i][0], 
                     final_results_eval[i][1], final_results_eval[i][2]))

    
    print("Elapsed training time: {}".format(end_time - start_time))
    
    # Saves training losses as a file
    

    with open(params.network_data_path + 'Training_losses_fold_' + str(k) + '_' + str(I) + 'trainedWith' + params.train_with + '_' + params.prep_step, 'w') as file:
        
        for i in range(len(losses)):
            
            if i != 0:
                
                file.write(',')
                
            file.write(str("%.20f" % losses[i]))
    
    
    # Plot losses and metrics and save their figures as .png
    
    # Loss
    
    plt.figure(figsize = (13,5))
    
    plt.plot(range(1,len(losses) + 1),losses), plt.xlabel('Iterations'), plt.ylabel(params.loss_fun + ' loss')
    
    plt.title('Evolution of ' + params.loss_fun + ' loss function, fold' + str(k))
    
    plt.savefig(params.network_data_path + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '_loss.png')
    
    # Metrics
    
    # Set metric lists to array
    
    train_metrics_array = np.array(train_metrics)
    
    eval_metrics_array = np.array(eval_metrics)


    
    # Save evaluation and training metrics to TXT files
    
    np.savetxt(params.network_data_path + 'ValidationMetrics_' + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '.txt', eval_metrics_array, fmt = '%s')
    
    np.savetxt(params.network_data_path + 'TrainingMetrics_' + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '.txt', eval_metrics_array, fmt = '%s')
    
    # Get unique metrics names
    
    metrics_unique = np.unique(train_metrics_array[:,0])
    
    # Apply a for loop for each metric type
    
    for metric_name in metrics_unique:
        
        metric_name = str(metric_name)
        
        ind_metric = np.where(train_metrics_array[:,0] == metric_name)[0]

        mean_metric_train = train_metrics_array[ind_metric,1]
        
        mean_metric_eval = eval_metrics_array[ind_metric,1]
        
        std_metric_train = train_metrics_array[ind_metric,2]
        
        std_metric_eval = eval_metrics_array[ind_metric,2]
        
        
        m_metric_train = []
        
        s_metric_train = []
        
        m_metric_eval = []
        
        s_metric_eval = []
        
        for i in range(len(mean_metric_train)):
            
            m_metric_train.append(float(mean_metric_train[i][:-1]))
            
            s_metric_train.append(float(std_metric_train[i][:-1]))
            
            m_metric_eval.append(float(mean_metric_eval[i][:-1]))
            
            s_metric_eval.append(float(std_metric_eval[i][:-1]))
        
        

        fig = plt.figure(figsize = (13,5))
        

        
        plt.errorbar(it, m_metric_train, yerr =  s_metric_train, color ='b', label = 'Training')
        
        plt.errorbar(it, m_metric_eval, yerr =  s_metric_eval, color = 'r', label = 'Validation')
        
        plt.xlabel('Iterations'), plt.ylabel(metric_name)
        
        plt.title('Evolution of ' + metric_name + ' fold ' + str(k))
        
        plt.legend()
        
        fig.savefig(params.network_data_path + '/' + 'trainedWith' + params.train_with + '_' + params.prep_step + '_fold_' + str(k) + '_' + metric_name + '.png')
    
    
    
    
    return losses[-1], final_results_train, final_results_eval, net.state_dict(), optimizer, optimizer.state_dict()

