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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from torchhd import functional, embeddings, bind, multiset, hard_quantize, permute, bundle

from scipy.special import softmax

from sklearn.cluster import KMeans, SpectralClustering

import time

# Model based on RegHD application for Single model regression -> No comparing which cluster

class Projection_2(nn.Module): 
    """Random Projection Class in case of changing as part of the encoding method"""

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self, in_features, out_features, requires_grad=False, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Projection_2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=requires_grad,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, 0, 1)
        self.weight.data.copy_(F.normalize(self.weight.data))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input * self.weight).as_subclass(MAP)

# Model based on RegHD application for Single model regression -> No comparing which cluster
class RegHD_AR(nn.Module):
    def __init__(self, size, d, models, number_ts, **kwargs):
        super(RegHD_AR, self).__init__()

        self.size = size # Past samples
        self.d = d # dimension of the hypervector
        self.dev = torch.device(kwargs['dev'])
        self.lr = torch.tensor([kwargs['opt'].learning_rate]).to(self.dev) # alpha
        self.opt = kwargs['opt']
        self.project = Projection_2(self.size, d, dtype=torch.float32, device=self.dev) # size features to d dimensions = hypervectors like weights?
        self.project.weight.data.normal_(0, 1) # Normal distributions mean=0.0, std=1.0
        self.bias = nn.parameter.Parameter(torch.empty(d, self.size), requires_grad=False).to(self.dev)
        self.bias.data.uniform_(0, 2 * math.pi) # bias
        self.kwargs = kwargs
        #self.covarianceMatrix = 0.1*torch.eye(self.size) # Only need 1 grid for covariance values in each sensor
        self.covarianceMatrix = torch.ones(self.d, self.d).to(self.dev)
        self.alpha = torch.zeros(1, d).type(torch.FloatTensor).to(self.dev) # Weight hypervector
        self.var = torch.tensor([0]).to(self.dev) # Variance in original samples
        self.gamma = torch.tensor([self.opt.alpha]).to(self.dev)
        self.current_ts = None
        self.updateCov = True
        self.past_sim = 0

    def hard_quantize(self, hv):
        """Function that returns the hapervector to the specified # of bits"""

        # Using positives and negatives numbers
        hv = ((hv)*(2**(self.opt.hd_representation-1)))/self.size
        hv = torch.tensor(hv // 1 + 2 ** (hv > 0) - 1, dtype = torch.int8)

        # Using only positives:
        # hv = (hv+self.size)/((self.size)*(2**(1-self.opt.hd_representation)))
        # hv = torch.floor(hv)

        return hv
    
    def encode(self, x, **kwargs): # encoding a single value TENSOR of size "size"
        enc = self.project(torch.reshape(x, (1, self.size)))
        enc = torch.cos(enc + self.bias) * torch.sin(enc) 
        #enc = self.hard_quantize(multiset(torch.transpose(enc, 0, 1)))
        enc = hard_quantize(torch.sum(enc, dim = 1))
        enc = torch.reshape(enc, (1, self.d))
        return enc

    def bind(self, x, y):
        return torch.logical_not(torch.logical_xor(x,y))

    def model_update(self, x, y, **kwargs): # update weights, variance and covariance matrix

        """if(self.current_ts != kwargs['ts']):
            self.current_ts = kwargs['ts']
            #self.covarianceMatrix = torch.ones(self.d, self.d) # Unique for each time series
            self.updateCov = True
            self.past_sim = 0"""

        model_result, enc = self(x, ts = kwargs['ts']) # Prediction

        self.var = (self.gamma * self.var) + ((1-self.gamma) * torch.var(x)) # MA for variance

        if (self.var[0] >= 0.001 or self.var[0] <= -0.001):
            
            # Update
            A_t = y - model_result # Innovation

            if (model_result < 0 or model_result > 1) or (abs(A_t) > 0.1):
                #temp = self.bind(self.covarianceMatrix > 0, enc > 0)
                #temp = torch.where(temp, abs(self.covarianceMatrix), -abs(self.covarianceMatrix))
                #const = hard_quantize(torch.sum(temp, dim=0))
                const = hard_quantize(torch.sum(torch.mul(self.covarianceMatrix,torch.transpose(enc, 0, 1)), dim = 1)) # Repeating value

                complete = torch.sum(const)
                G_t = const / (complete + (self.var*self.d)) # Kalman Gain
                self.alpha += G_t*A_t*self.lr
                #self.covarianceMatrix += hard_quantize(torch.mul(G_t, torch.reshape(const, (self.d, 1))))
                #x = torch.mul(G_t, torch.reshape(const, (self.d, 1)))
                #inter = self.bind(G_t > 0, torch.reshape(const > 0, (self.d, 1))).cuda()
                #self.covarianceMatrix += torch.where(inter, 1, -1)
                self.covarianceMatrix += bind(hard_quantize(G_t), torch.reshape(const, (self.d, 1)))
    
    def forward(self, x, **kwargs): # With weights x: array of values compute the prediction
        #x = torch.tensor(x.reshape((self.size, 1)), dtype = torch.float32)    
        enc = self.encode(x)
        model_result = torch.sum(torch.mul(enc, self.alpha))
        # prediction
        
        return model_result, enc

    def train(self, sets_training, matrix_1_norm, matrix_1_norm_org, y, epochs, sets_cv):

        size = matrix_1_norm.shape[0]
        matrix_1_norm = torch.tensor(matrix_1_norm, dtype = torch.float32, device=self.dev)
        y = torch.tensor(y, dtype = torch.float32, device=self.dev)
        
        for _ in range(epochs): # Number of iterations for all the samples set to 1
            
            for n in tqdm(range(size)): # For each ts
                samples = matrix_1_norm[n, :]
            
                for i in (sets_training): # For each set in the rolling window
                    
                    sample = samples[i:i+self.size] 
                    label = samples[i+self.size]

                    self.model_update(sample, label, ts = n, time = i) # Pass input and label to train
                    
                    predictions_testing, enc = self(sample, ts = n)
                    y[n, i+self.size] = predictions_testing

                    """if (i % 5000 == 0):
                        pred, labels_full = self.test(sets_cv, matrix_1_norm, matrix_1_norm_org, y)
                        print(f"\nCross Validation root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")"""
                
                if (n % self.opt.print_freq == 0 and n != 0):
                    pred, labels_full = self.test(sets_cv, matrix_1_norm, matrix_1_norm_org, y.cpu())
                    print(f"\nCross Validation root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
    
    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv = True):
        matrix_1_norm_org = torch.tensor(matrix_1_norm_org, dtype = torch.float32, device=self.dev)
        if (not cv):
            matrix_1_norm = torch.tensor(matrix_1_norm, dtype = torch.float32, device=self.dev)
        pred = []
        labels_full = []
        for n in range(matrix_1_norm.shape[0]): # For each ts
            samples = matrix_1_norm[n, :]
        
            for i in (sets_testing): # For each set in the rolling window
                
                sample = samples[i:i+self.size] 
                #label = torch.tensor(samples[i+self.size])

                # Pass samples from test to model (forward function)
                predictions, enc = self(sample, ts = n)
                pred.append(float(predictions))
                y[n, i+self.size] = float(predictions)
                labels_full.append(matrix_1_norm_org[n, i+self.size].cpu())
        if (not cv):
            error = mean_squared_error(labels_full, pred, squared=False)
            print(
            f"Testing root mean squared error of testing {(error):.3f}")
            return error
        
        return pred, labels_full


def Return_Model(size, d, models, number_ts, opt, dev):

    torch.random.manual_seed(opt.trial)

    model_hd = RegHD_AR(size, d, models, number_ts, opt = opt, dev = dev)  # 1 class, 5

    return model_hd