# For RegHD
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchmetrics
from tqdm import tqdm
from torchhd.map import MAP

import numpy as np
# Model based on RegHD application for Single model regression -> No comparing which cluster

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from torchhd import functional, embeddings, cos_similarity, bind, multiset, hard_quantize, permute, bundle

from scipy.special import softmax

from sklearn.cluster import KMeans, SpectralClustering

from numpy.linalg import inv

# Model based on RegHD application for Single model regression -> No comparing which cluster
class RegHD_AR(nn.Module):
    def __init__(self, size, d, models, number_ts, **kwargs):
        super(RegHD_AR, self).__init__()

        # Define the parameters of the Kalman filter
        self.state_dim = size # Dimension of the state vector (weights)
        self.obs_dim = 1 # Dimension of the observation vector (current value of the time series)
        self.A = torch.eye(self.state_dim) # State transition matrix (identity matrix)
        self.C = torch.zeros((self.obs_dim, self.state_dim)) # Observation matrix
        self.C[0, 0] = 1 # We want to predict the first component of the state vector (i.e., the current value)
        self.covarianceMatrix = {}
        for i in range(number_ts):
            self.covarianceMatrix[i] = 0.1*torch.eye(self.state_dim) # Process noise covariance matrix
        self.R = 0.1*torch.eye(self.obs_dim) # Measurement noise covariance matrix
        """self.alpha = {}
        for i in range(number_ts):
            self.alpha[i] = torch.zeros((self.state_dim, 1)) # Initial state estimate"""
        self.alpha = torch.zeros((self.state_dim, 1))
        self.P = torch.eye(self.state_dim) # Initial error covariance matrix
        self.opt = kwargs['opt']
        self.var = 0
    
    def model_update(self, x, y, **kwargs): # update # y = no hv
        model_result = self(x, ts = kwargs['ts'])
        x = torch.reshape(torch.tensor(x, dtype = torch.float32), (1, self.state_dim))
        const = torch.matmul(x, torch.matmul(self.covarianceMatrix[kwargs['ts']], torch.transpose(x, 0, 1)))
        if (float(const + torch.var(x)) >= 0.001 or float(const + torch.var(x)) <= -0.001):
            G_t = torch.matmul(self.covarianceMatrix[kwargs['ts']], torch.transpose(x, 0, 1)) / (const + torch.var(x))
    
            A_t = y - model_result
        #for i in range(kwargs['ts'], len(self.alpha)):
            self.alpha += G_t*A_t
            #self.alpha[kwargs['ts']] += G_t*A_t
            self.covarianceMatrix[kwargs['ts']] -= torch.matmul(torch.matmul(G_t, x), self.covarianceMatrix[kwargs['ts']])
        
    
    def forward(self, x, **kwargs): # With weights x: array of values
        #x = torch.tensor(x.reshape((self.size, 1)), dtype = torch.float32)

        #model_result = torch.matmul(torch.tensor(x, dtype=torch.float32), self.alpha[kwargs['ts']])
        model_result = torch.matmul(torch.tensor(x, dtype=torch.float32), self.alpha)
        #return model_result, enc, hvs
        if(model_result > 10 or model_result < -10):
            print("wrong")
            """model_result = 0
        if(model_result > 1):
            model_result = 1"""
        return model_result

    def train(self, sets_training, matrix_1_norm, matrix_1_norm_org, y, epochs, sets_cv):

        for _ in range(epochs): # Number of iterations for all the samples
            
            for n in tqdm(range(matrix_1_norm.shape[0])):
                samples = matrix_1_norm[n, :]
            
                for i in (sets_training):
                    
                    sample = samples[i:i+self.state_dim]
                    """if np.isnan(samples[i+self.size]):
                        predictions_testing, enc, hvs = self(sample, ts = n)
                        samples[i+self.size] = predictions_testing"""
                    label = torch.tensor(samples[i+self.state_dim])

                    self.model_update(sample, label, ts = n, time = i) # Pass input and label to train
                    
                    predictions_testing = self(sample, ts = n)
                    y[n, i+self.state_dim] = float(predictions_testing)
                
                if (n % self.opt.print_freq == 0):
                    pred, labels_full = self.test(sets_cv, matrix_1_norm, matrix_1_norm_org, y)
                    print(f"\nCross Validation root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
                    #print(f"Self.var = {self.var}")
    
    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv = True):
        pred = []
        labels_full = []
        for i in (sets_testing):
            samples = matrix_1_norm_org[:, i:i+self.state_dim]
            labels = matrix_1_norm[:, i+self.state_dim]
            for n in range(samples.shape[0]):
                sample = samples[n, :]
                if(np.isnan(labels[n])):
                    predictions = self(sample, ts = n)
                    matrix_1_norm[n, i+self.state_dim] = predictions
                label = torch.tensor(labels[n])
                # Pass samples from test to model (forward function)
                predictions = self(sample, ts = n)
                pred.append(float(predictions))
                y[n, i+self.state_dim] = float(predictions)
                labels_full.append(matrix_1_norm_org[n, i+self.state_dim])
        if (not cv):
            error = mean_squared_error(labels_full, pred, squared=False)
            print(
            f"Testing root mean squared error of testing {(error):.3f}")
            return error
        
        return pred, labels_full


def Return_Model(size, d, models, number_ts, opt):

    model_hd = RegHD_AR(size, d, models, number_ts, opt = opt)  # 1 class, 5

    return model_hd


