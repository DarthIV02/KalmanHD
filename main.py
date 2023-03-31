import os
import sys
import argparse
import random
import numpy as np
from Stuff.DatasetLoader import DatasetLoader
from Stuff.Initializer import Initializer

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')
    
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate gradient for the model')
    
    parser.add_argument('--hd_encoder', type=str, default='nonlinear',
                        choices=['nonlinear', 'time_encoding', 'bind_timeseries'],
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
    
    parser.add_argument('--num_timestamp', type=int, default=15, 
                        help='Number of timestamps used for training and testing over all the timeseries')
    
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')
    
    parser.add_argument('--model', type=str, default='DNN',  # CHAAAAANGE
                        choices=['RegHD', 'VAE', 'DNN'],
                        help='Model to test')
    
    parser.add_argument('--size_of_sample', type=int, default=20, 
                        help='Number of previous samples before forecasting')
    
    parser.add_argument('--levels', type=int, default=6, 
                        help='Number of levels to divide encoding when using a different hd-encoder')
    
    parser.add_argument('--retraining', type=bool, default=True, # CHANGEEEE
                        help='If the model with this particular data, has been previously trained, set retraining = True')
    
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
    sets = np.random.choice(matrix_1_norm.shape[1]-40, opt.num_timestamp, replace=False)
    sets_training, sets_testing = sets[:int(len(sets)*.8)], sets[int(len(sets)*.8):]

    if opt.model == "RegHD":
        from models.RegHD.RegHD import Return_Model
        model = Return_Model(opt.size_of_sample, opt.dimension_hd, opt.models, opt)
        model.train(sets_training, matrix_1_norm, opt.epochs)
        model.test(sets_testing, matrix_1_norm)

    if opt.model == "DNN":
        from models.DNN.DNN import Return_Model, Train_Model, Test_Model
        model = Return_Model(opt.size_of_sample)
        model = Train_Model(model, matrix_1_norm, sets_training, opt.retraining, opt.dataset, opt.size_of_sample, opt.epochs)
        model, dif_dnn = Test_Model(model, matrix_1_norm, sets_testing, opt.size_of_sample)        

    if opt.model == "VAE":
       from models.VAE.VAE import Return_Model, Train_Model, Test_Model
       vae, enc, dec, es = model = Return_Model(opt.size_of_sample)
       vae, enc, dec, es = Train_Model(vae, es, matrix_1_norm, sets_training, opt.retraining, opt.dataset, opt.size_of_sample + 1, opt.epochs)
       vae, dif_vae = Test_Model(vae, matrix_1_norm, sets_testing, opt.size_of_sample)   

if __name__ == '__main__':
    main()
