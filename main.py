import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from Stuff.DatasetLoader import DatasetLoader
from Stuff.Initializer import Initializer
import matplotlib.pyplot as plt

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')
    
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate gradient for the model')
    
    parser.add_argument('--hd_encoder', type=str, default='nonlinear',
                        choices=['nonlinear', 'time_encoding', 'bind_timeseries', 'linear'],
                        help='the type of hd encoding function to use')
    
    parser.add_argument('--clustering', type=str, default='none',
                        choices=['none', 'spectral_clustering', 'kmeans'],
                        help='the type of of clustering method to incorporate')
    
    parser.add_argument('--models', type=int, default=1, 
                        help='When using clustering, the number of models to seperate the clustering')
    
    parser.add_argument('--dimension_hd', type=int, default=10000,
                        help='number of dimensions in the hypervector')
    
    parser.add_argument('--dataset', type=str, default='SanFranciscoTraffic', 
                        choices=['SanFranciscoTraffic', 'MetroInterstateTrafficVolume', 
                                 'GuangzhouTraffic', 'EnergyConsumptionFraunhofer', 'ElectricityLoadDiagrams'],
                        help='Dataset to initialize')
    
    #parser.add_argument('--num_timestamp', type=int, default=15, 
                        #help='Number of timestamps in each rolling window used for training and testing over all the timeseries')
    
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')
    
    parser.add_argument('--model', type=str, default='RegHD', 
                        choices=['RegHD', 'VAE', 'DNN'],
                        help='Model to test')
    
    parser.add_argument('--size_of_sample', type=int, default=20, 
                        help='Number of previous samples before forecasting')
    
    parser.add_argument('--levels', type=int, default=6, 
                        help='Number of levels to divide encoding when using a different hd-encoder')
    
    parser.add_argument('--retraining', type=bool, default=False,
                        help='If the model with this particular data, has been previously trained, set retraining = True')
    
    parser.add_argument('--add_weights', type=str, default='Kalman Filter', 
                        choices=['false', 'Yule Walker', 'Kalman Filter'],
                        help='If adding weights, choose method')
    
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
    #sets = np.random.choice(matrix_1_norm.shape[1]-40, opt.num_timestamp, replace=False)
    #sets_training, sets_testing = sets[:int(len(sets)*.8)], sets[int(len(sets)*.8):]
    sets_training = [i for i in range(int((matrix_1_norm.shape[1]-opt.size_of_sample)*0.7))]
    print(len(sets_training))
    sets_testing = [i for i in range(int((matrix_1_norm.shape[1]-opt.size_of_sample)*0.7), int((matrix_1_norm.shape[1]-opt.size_of_sample)*0.9))]
    sets_cv = [i for i in range(int((matrix_1_norm.shape[1]-opt.size_of_sample)*0.9), (matrix_1_norm.shape[1]-opt.size_of_sample))]

    if opt.add_weights == "Kalman Filter":
        from models.RegHD.RegHD_AR import Return_Model
        model = Return_Model(opt.size_of_sample, opt.dimension_hd, opt.models, matrix_1_norm.shape[0], opt)
        y = np.zeros((matrix_1_norm.shape))
        model.train(sets_training, matrix_1_norm, y, opt.epochs, sets_cv)
        #y, label = model.test2(sets_testing[0], matrix_1_norm, len(sets_testing))
        model.test(sets_testing, matrix_1_norm, y, cv=False)

        x = [i+opt.size_of_sample for i in sets_training]
        x_2 = [i+opt.size_of_sample for i in sets_testing]

        print(x[-1], x_2[0])

        for i in range(matrix_1_norm.shape[0]):
            """ result_dict = []

            for n in range(label.shape[1]):
                result_dict.append({'timestamp': sets_testing[0] + n, 'pred': y[i, n+opt.size_of_sample], 'label':label[i, n]})
            df = pd.DataFrame.from_dict(result_dict) 
            df.to_csv (f'results/SanFranciscoTraffic_csv/time_series_{i}.csv', index=False, header=True) """

            plt.plot(x, y[i, x[0]:x[-1]+1], 'b')
            plt.plot(x_2, y[i, x_2[0]:x_2[-1]+1], 'r')
            plt.plot(x+x_2, matrix_1_norm[i, x[0]:x_2[-1]+1], "k")
            plt.title('Predictions')
            plt.xlabel('Timestamp')
            plt.ylabel('Prediction')
            plt.savefig(f'results/SanFranciscoTraffic_img/RegHD_AR_3/time_series_{i}.png')
            plt.clf()


    elif opt.model == "RegHD":
        from models.RegHD.RegHD import Return_Model
        model = Return_Model(opt.size_of_sample, opt.dimension_hd, opt.models, matrix_1_norm.shape[0], opt)
        y = np.zeros((matrix_1_norm.shape))
        model.train(sets_training, matrix_1_norm, y, opt.epochs, sets_cv)
        #y, label = model.test2(sets_testing[0], matrix_1_norm, len(sets_testing))
        model.test(sets_testing, matrix_1_norm, y, cv=False)

        # Save results

        x = [i+opt.size_of_sample for i in sets_training]
        x_2 = [i+opt.size_of_sample for i in sets_testing]

        print(x[-1], x_2[0])

        for i in range(matrix_1_norm.shape[0]):
            """ result_dict = []

            for n in range(label.shape[1]):
                result_dict.append({'timestamp': sets_testing[0] + n, 'pred': y[i, n+opt.size_of_sample], 'label':label[i, n]})
            df = pd.DataFrame.from_dict(result_dict) 
            df.to_csv (f'results/SanFranciscoTraffic_csv/time_series_{i}.csv', index=False, header=True) """

            plt.plot(x, y[i, x[0]:x[-1]+1], 'b')
            plt.plot(x_2, y[i, x_2[0]:x_2[-1]+1], 'r')
            plt.plot(x+x_2, matrix_1_norm[i, x[0]:x_2[-1]+1], "k")
            plt.title('Predictions')
            plt.xlabel('Timestamp')
            plt.ylabel('Prediction')
            plt.savefig(f'results/SanFranciscoTraffic_img/AR/time_series_{i}.png')
            plt.clf()

    if opt.model == "DNN":
        from models.DNN.DNN import Return_Model, Train_Model, Test_Model
        model = Return_Model(opt.size_of_sample)
        model = Train_Model(model, matrix_1_norm, sets_training, opt.retraining, opt.dataset, opt.size_of_sample, opt.epochs)
        model, dif_dnn = Test_Model(model, matrix_1_norm, sets_testing, opt.size_of_sample)        

    if opt.model == "VAE":
       from models.VAE.VAE import Return_Model, Train_Model, Test_Model
       vae, enc, dec, es = model = Return_Model(opt.size_of_sample + 1)
       vae, enc, dec, es = Train_Model(vae, es, matrix_1_norm, sets_training, opt.retraining, opt.dataset, opt.size_of_sample + 1, opt.epochs)
       vae, dif_vae = Test_Model(vae, matrix_1_norm, sets_testing, opt.size_of_sample + 1)   

if __name__ == '__main__':
    main()
