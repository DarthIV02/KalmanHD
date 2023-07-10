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

import time

class Projection_2(nn.Module):

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

        self.size = size
        self.d = d
        self.lr = kwargs['opt'].learning_rate # alpha
        """self.covarianceMatrix = {}
        for i in range(number_ts):
            self.covarianceMatrix[i] = torch.eye(size)"""
        #self.covarianceMatrix = torch.eye(size)
        self.M = torch.zeros(d).float()
        self.opt = kwargs['opt']
        self.project = Projection_2(self.size, d, dtype=torch.float32) # 5 features, 10000 dimensions = hypervectors like weights?
        self.project.weight.data.normal_(0, 1) # Normal distributions mean=0.0, std=1.0
        self.bias = nn.parameter.Parameter(torch.empty(d, self.size), requires_grad=False)
        self.bias.data.uniform_(0, 2 * math.pi) # bias
        self.kwargs = kwargs
        """self.alpha = {}
        for i in range(number_ts):
            #model_hd.alpha[i] = np.random.rand(size)
            self.alpha[i] = torch.ones(size)"""
        #self.alpha = torch.zeros((size))
        self.covarianceMatrix = {}
        for i in range(number_ts):
            self.covarianceMatrix[i] = 0.1*torch.eye(self.size) # Process noise covariance matrix
        self.alpha = {}
        for i in range(number_ts):
            self.alpha[i] = torch.rand((1, self.size)) # Initial state estimate
        self.var = 0


    def encode(self, x, **kwargs): # encoding a single value TENSOR
        #x_2 = x * self.alpha[kwargs['ts']]
        enc = self.project(torch.reshape(x, (1, self.size)))
        enc = torch.cos(enc + self.bias) * torch.sin(enc) 
        enc_2 = enc * self.alpha[kwargs['ts']]
        return functional.hard_quantize(multiset(torch.transpose(enc_2, 0, 1))), torch.transpose(enc, 0, 1)
        #return functional.hard_quantize(enc)
        #return multiset(enc)
    
    def model_update(self, x, y, **kwargs): # update # y = no hv
        model_result, enc, hvs = self(x, ts = kwargs['ts'])
        #if (np.isnan(y)):
            #y = model_result
        x = torch.reshape(torch.tensor(x, dtype = torch.float32), (1, self.size))
        #const = torch.matmul(x, torch.matmul(self.covarianceMatrix[kwargs['ts']], torch.transpose(x, 0, 1)))
        """const = 0
        for i in range(self.size):
            for j in range(self.size):
                const += self.covarianceMatrix[kwargs['ts']][i, j] * F.linear(hvs[i], hvs[j])"""
        matrix_const = torch.matmul(hvs, torch.transpose(hvs, 0, 1)) * self.covarianceMatrix[kwargs['ts']]
        const = torch.sum(matrix_const)
        self.var = (self.opt.alpha * self.var) + (1-self.opt.alpha) * torch.var(hvs)
        const[float(const + self.var) <= 0.01 and float(const + self.var) >= -0.01] = 1
        if ((float(const + self.var) >= 0.01 or float(const + self.var) <= -0.01) and float(torch.norm(self.M))!=0):
            G_t = torch.matmul(self.covarianceMatrix[kwargs['ts']], hvs) / (const + self.var)
            
            #A_t = y - model_result
            #y_enc = self.project(torch.reshape(y, (1, 1)).type(torch.FloatTensor))
            #y_enc = torch.cos(y_enc + self.bias) * torch.sin(y_enc) 
            #A_t = 1 - cos_similarity(functional.hard_quantize(multiset(y_enc)), enc)
            M_inverse = self.M / (torch.norm(self.M)**2)
            A_t = bind(functional.hard_quantize(M_inverse * y), enc)
            A_t[A_t==1] = 0
        #for i in range(kwargs['ts'], len(self.alpha)):
            #self.alpha[i] += G_t*A_t
            #self.alpha[kwargs['ts']] += G_t*A_t
            self.alpha[kwargs['ts']] += torch.matmul(G_t, A_t)
            self.covarianceMatrix[kwargs['ts']] -= torch.matmul(torch.matmul(G_t, torch.transpose(hvs, 0, 1)), self.covarianceMatrix[kwargs['ts']])
        update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
        self.M = update # New
        
    
    def forward(self, x, **kwargs): # With weights x: array of values
        x = torch.tensor(x.reshape((self.size, 1)), dtype = torch.float32)
        #x = torch.tensor(x, dtype = torch.float32)
        enc, hvs = self.encode(x, ts = kwargs['ts'])
        #enc = self.encode(x, ts = kwargs['ts'])
        #enc = torch.reshape(enc, (1, self.d))

        model_result = F.linear(enc.type(torch.FloatTensor), self.M.type(torch.FloatTensor))
        #model_result = torch.sum(enc)
        #return model_result, enc, hvs
        return model_result, enc, hvs

    def train(self, sets_training, matrix_1_norm, matrix_1_norm_org, y, epochs, sets_cv):

        for _ in range(epochs): # Number of iterations for all the samples
            
            for n in tqdm(range(matrix_1_norm.shape[0])):
                samples = matrix_1_norm[n, :]
            
                for i in (sets_training):
                    
                    sample = samples[i:i+self.size]
                    """if np.isnan(samples[i+self.size]):
                        predictions_testing, enc, hvs = self(sample, ts = n)
                        samples[i+self.size] = predictions_testing"""
                    label = torch.tensor(samples[i+self.size])

                    self.model_update(sample, label, ts = n, time = i) # Pass input and label to train
                    
                    predictions_testing, enc, hvs = self(sample, ts = n)
                    y[n, i+self.size] = float(predictions_testing)
                
                if (n % self.opt.print_freq == 0):
                    pred, labels_full = self.test(sets_cv, matrix_1_norm, matrix_1_norm_org, y)
                    print(f"\nCross Validation root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
                    print(f"Self.var = {self.var}")
    
    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv = True):
        pred = []
        labels_full = []
        for i in (sets_testing):
            samples = matrix_1_norm_org[:, i:i+self.size]
            labels = matrix_1_norm[:, i+self.size]
            for n in range(samples.shape[0]):
                sample = samples[n, :]
                if(np.isnan(labels[n])):
                    predictions, enc, hvs = self(sample, ts = n)
                    matrix_1_norm[n, i+self.size] = predictions
                label = torch.tensor(labels[n])
                # Pass samples from test to model (forward function)
                predictions, enc, hvs = self(sample, ts = n)
                pred.append(float(predictions))
                y[n, i+self.size] = float(predictions)
                labels_full.append(matrix_1_norm_org[n, i+self.size])
        if (not cv):
            print(
            f"Testing root mean squared error of testing {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
        
        return pred, labels_full


def Return_Model(size, d, models, number_ts, opt):

    torch.random.manual_seed(opt.trial)

    model_hd = RegHD_AR(size, d, models, number_ts, opt = opt)  # 1 class, 5

    return model_hd