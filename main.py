import os
import sys
import argparse
import random
import numpy as np
from Stuff.DatasetLoader import DatasetLoader

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
                        choices=['none', 'spectral_clustering', 'k_means'],
                        help='the type of of clustering method to incorporate')
    parser.add_argument('--models', type=int, default=1, 
                        help='When using clustering, the number of models to seperate the clustering')
    parser.add_argument('--dimension', type=int, default=10000,
                        help='number of dimensions in the hypervector')
    parser.add_argument('--dataset', type=str, default='SanFranciscoTraffic', 
                        choices=['SanFranciscoTraffic', 'MetroInterstateTrafficVolume', 
                                 'GuangzhouTraffic', 'EnergyConsumptionFraunhofer', 'ElectricityLoadDiagrams'],
                        help='Dataset to initialize')
    parser.add_argument('--num_timestamp_training', type=int, default=12, 
                        help='Number of timestamps used for training over all the timeseries')
    parser.add_argument('--num_timestamp_testing', type=int, default=3, 
                        help='Number of timestamps used for testing over all the timeseries')
    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')
    
    parser.add_argument('--model', type=str, default='RegHD', 
                        choices=['RegHD', 'VAE', 'DNN'],
                        help='Model to test')
    
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

    if opt.model == "RegHD":

        from models.RegHD import RegHD
        Trainer = BasicHD()
        Trainer.start()
        print("===RegHD Complete===")

if __name__ == '__main__':
    main()
