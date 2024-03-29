import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from Stuff.DatasetLoader import DatasetLoader
from Stuff.Initializer import Initializer
import matplotlib.pyplot as plt
import csv
import torch

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1000,
                        help='print frequency')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate gradient for the model')
    
    parser.add_argument('--hd_encoder', type=str, default='nonlinear',
                        choices=['nonlinear', 'time_encoding', 'bind_timeseries', 'linear'],
                        help='the type of hd encoding function to use')
    
    parser.add_argument('--hd_representation', type=int, default=1,
                        help='Number of bits to use for the hypervector representation')
    
    parser.add_argument('--models', type=int, default=1,
                        help='When using clustering, the number of models to seperate the clustering')
    
    parser.add_argument('--dimension_hd', type=int, default=1000,
                        help='number of dimensions in the hypervector')
    
    parser.add_argument('--dataset', type=str, default='SanFranciscoTraffic', 
                        choices=['SanFranciscoTraffic', 'MetroInterstateTrafficVolume', 
                                 'GuangzhouTraffic', 'EnergyConsumptionFraunhofer', 'ElectricityLoadDiagrams'],
                        help='Dataset to initialize')
    
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')
    
    parser.add_argument('--novelty', type=float, default=0.2,
                        help='cosine similarity difference for a timeseries to be considered new')
    
    parser.add_argument('--model', type=str, default='KalmanHD', 
                        choices=['RegHD', 'VAE', 'DNN', 'KalmanFilter', 'KalmanHD'],
                        help='Model to test')
    
    parser.add_argument('--size_of_sample', type=int, default=20, 
                        help='Number of previous samples before forecasting')
    
    parser.add_argument('--levels', type=int, default=6, 
                        help='Number of levels to divide encoding when using a linear hd-encoder')
    
    parser.add_argument('--retraining', type=bool, default=False,
                        help='If the model with this particular data, has been previously trained, set retraining = True')
    
    parser.add_argument('--s', type=int, default=5, 
                            help='Number of consecutive missing values')
    
    parser.add_argument('--p', type=float, default=0.0, 
                        help='Percentage of probability of value missing')
    
    parser.add_argument('--alpha', type=float, default=0.3, 
                        help='Percentage of moving average for variance')

    parser.add_argument('--gaussian_noise', type=float, default=0.0, 
                        help='Standard deviation of the gaussian noise')
    
    parser.add_argument('--poisson_noise', type=float, default=0.0, 
                        help='Lambda for the poisson noise')
    
    parser.add_argument('--device', type=str, default='gpu', 
                        choices=['cpu', 'gpu'],
                        help='Device to use, either cpu or gpu')
    
    #parser.add_argument('--flipping_rate', type=float, default=0.0, 
    #                    help='Percentage of bits flipped')

    opt = parser.parse_args()

    return opt

def main():
    opt = parse_option()
    
    print("============================================")
    print(opt)
    print("============================================")

    # set seed for reproducing
    random.seed(opt.trial)
    np.random.seed(opt.trial)

    # Set data loader
    dl = DatasetLoader(opt.dataset)

    matrix_1_original = dl.dataset_load_and_preprocess("original")
    matrix_1_norm = dl.dataset_load_and_preprocess("normalized")
    matrix_1_norm_org = np.copy(matrix_1_norm)
    
    # rolling window values for training, testing and cross validation
    sets_training = [i for i in range(int((matrix_1_norm.shape[1]-opt.size_of_sample)*0.7))]
    print(len(sets_training))
    print(matrix_1_norm.shape)
    sets_testing = [i for i in range(int((matrix_1_norm.shape[1]-opt.size_of_sample)*0.7), int((matrix_1_norm.shape[1]-opt.size_of_sample)*0.9))]
    sets_cv = [i for i in range(int((matrix_1_norm.shape[1]-opt.size_of_sample)*0.9), (matrix_1_norm.shape[1]-opt.size_of_sample))]
    sets_missing = {}
    for i in range(matrix_1_norm.shape[0]):
        sets_missing[i] = []

    #add missing values
    for i in range(0, matrix_1_norm.shape[1], opt.s):
        for j in range(matrix_1_norm.shape[0]):
            if(random.random() < opt.p):
                matrix_1_norm[j, i:i+opt.s] = 0
                sets_missing[j].append((i, i+opt.s-1))

    #add gaussian noise
    gaussian_noise = np.random.normal(0, opt.gaussian_noise, size=(matrix_1_norm.shape[0], matrix_1_norm.shape[1]))
    matrix_1_norm += gaussian_noise

    if opt.poisson_noise != 0:
        poisson_noise = np.random.poisson(lam=opt.poisson_noise, size=(matrix_1_norm.shape[0], matrix_1_norm.shape[1]))
        matrix_1_norm += poisson_noise

    # Use CPU
    if opt.device == "cpu":
        torch.backends.cudnn.enabled = False
        torch.cuda.is_available = lambda : False

    if torch.cuda.is_available():
        print("Using GPU device")
        dev = "cuda:0"
    else:
        print("Using CPU device")
        dev = "cpu"
    
    # File to save the results
    csv_file = 'results_poisson2.csv'

    if opt.model == "RegHD":
        from models.RegHD.RegHD import Return_Model
        model = Return_Model(opt.size_of_sample, opt.dimension_hd, opt.models, matrix_1_norm.shape[0], opt, dev)
        y = np.zeros((matrix_1_norm.shape))
        model.train(sets_training, matrix_1_norm, matrix_1_norm_org, y, opt.epochs, sets_cv)
        error = model.test(sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv=False)

    if opt.model == "KalmanFilter":
        from models.KalmanFilter.AR import Return_Model
        model = Return_Model(opt.size_of_sample, opt.dimension_hd, opt.models, matrix_1_norm.shape[0], opt)
        y = np.zeros((matrix_1_norm.shape))
        model.train(sets_training, matrix_1_norm, matrix_1_norm_org, y, opt.epochs, sets_cv)
        error = model.test(sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv=False)

    if opt.model == "KalmanHD":
        #from models.ARHD.RegHD_AR_M import Return_Model
        from models.KalmanHD.KalmanHD_binary import Return_Model
        model = Return_Model(opt.size_of_sample, opt.dimension_hd, opt.models, matrix_1_norm.shape[0], opt, dev)
        y = np.zeros((matrix_1_norm.shape))
        model.train(sets_training, matrix_1_norm, matrix_1_norm_org, y, opt.epochs, sets_cv)
        error = model.test(sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv=False)

    if opt.model == "DNN":
        from models.DNN.DNN import Return_Model, Train_Model, Test_Model
        model = Return_Model(opt.size_of_sample)
        model = Train_Model(model, matrix_1_norm, sets_training, opt.retraining, opt.dataset, opt.size_of_sample, opt.epochs)
        error = Test_Model(model, matrix_1_norm_org, sets_testing, opt.size_of_sample)        

    if opt.model == "VAE":
       from models.VAE.VAE import Return_Model, Train_Model, Test_Model
       vae, enc, dec, es = model = Return_Model(opt.size_of_sample + 1)
       vae, enc, dec, es = Train_Model(vae, es, matrix_1_norm, sets_training, opt.retraining, opt.dataset, opt.size_of_sample + 1, opt.epochs)
       error = Test_Model(vae, matrix_1_norm_org, sets_testing, opt.size_of_sample + 1)  

    # Write results
    
    add_value_to_csv(csv_file, opt.dataset, opt.model, opt.models, opt.novelty, opt.learning_rate, opt.hd_representation, "Poisson", opt.poisson_noise, error)

    # Visualize results

    """x = [i+opt.size_of_sample for i in sets_training]
    x_2 = [i+opt.size_of_sample for i in sets_testing]

    print(x[-1], x_2[0])

    for i in range(matrix_1_norm.shape[0]):
        result_dict = []

        plt.plot(x, y[i, x[0]:x[-1]+1], 'b')
        plt.plot(x_2, y[i, x_2[0]:x_2[-1]+1], 'r')
        plt.plot(x+x_2, matrix_1_norm_org[i, x[0]:x_2[-1]+1], "k")
        for x1, x2 in sets_missing[i]:
            if x1+opt.size_of_sample <= x_2[-1] - opt.s:
                plt.axvspan(x1+opt.size_of_sample, x2+opt.size_of_sample, color='black', alpha=0.2)
        plt.title('Predictions')
        plt.xlabel('Timestamp')
        plt.ylabel('Prediction')
        plt.savefig(f'results2/EnergyConsumptionFraunhofer/AR_Debug/time_series_{i}.png')
        plt.clf()""" 

def add_value_to_csv(csv_file, ts, model, models, novelty, lr, hd_bites, noise, noiseVol, mae):
    # Check if the CSV file exists
    try:
        with open(csv_file, 'r') as file:
            # Read the existing data from the CSV file
            reader = csv.reader(file)
            data = list(reader)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one with the header
        data = [['TimeSeries Dataset', 'Model', 'Models', 'Novelty', 'Lr', 'hd_bites','Noise', 'Level Noise', 'MAE']]
    
    # Add the value to the data
    data.append([ts, model, models, novelty, lr, hd_bites, noise, noiseVol, mae])
    
    # Write the updated data back to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == '__main__':
    main()
