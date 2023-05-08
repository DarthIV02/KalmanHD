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


    def encode(self, x, **kwargs): # encoding a single value TENSOR
        #x_2 = x * self.alpha[kwargs['ts']]
        enc = self.project(torch.reshape(x, (1, self.size)))
        enc = torch.cos(enc + self.bias) * torch.sin(enc) 
        enc_2 = enc * self.alpha[kwargs['ts']]
        return functional.hard_quantize(multiset(torch.transpose(enc_2, 0, 1))), torch.transpose(enc, 0, 1)
        #return functional.hard_quantize(enc)
        #return multiset(enc)
    
    def model_update(self, x, y, **kwargs): # update # y = no hv
        #model_result, enc, hvs = self(x, ts = kwargs['ts'])
        """model_result, enc = self(x, ts = kwargs['ts'])
        x = torch.reshape(torch.tensor(x, dtype = torch.float32), (1, self.size))
        const = (F.linear(x, torch.transpose(F.linear(self.covarianceMatrix, x),0,1)))
        G_t = F.linear(self.covarianceMatrix, x) / (const + torch.var(x))"""
        #y_enc = self.project(torch.reshape(y.float(), (1,1)))
        #y_enc = torch.cos(y_enc + self.bias) * torch.sin(y_enc) 
        #M_inverse = self.M / (torch.norm(self.M)**2)
        #A_t = 1 - cos_similarity(functional.hard_quantize(M_inverse * y), enc)
        #A_t = y - model_result
        #A_t = bind(functional.hard_quantize(M_inverse * y), enc)
        #A_t = 1 - cos_similarity(self.encode(y), enc)
        """update = self.M + (float(self.lr) * float(y - model_result) * enc)
        if float(cos_similarity(update, self.M)) >= 1-0.0001:
            self.alpha[kwargs['ts']] += torch.reshape(torch.transpose(G_t, 0, 1)*A_t, (self.size, ))
        else:
            update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
            self.M = update # New """
        """self.alpha += torch.reshape(torch.transpose(G_t, 0, 1)*A_t, (self.size, ))
        self.alpha += torch.matmul(G_t, A_t)
        update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
        self.M = update # New
        self.covarianceMatrix -= torch.matmul(torch.matmul(G_t, x), self.covarianceMatrix)"""
        model_result, enc, hvs = self(x, ts = kwargs['ts'])
        x = torch.reshape(torch.tensor(x, dtype = torch.float32), (1, self.size))
        #const = torch.matmul(x, torch.matmul(self.covarianceMatrix[kwargs['ts']], torch.transpose(x, 0, 1)))
        const = 0
        for i in range(self.size):
            for j in range(self.size):
                const += self.covarianceMatrix[kwargs['ts']][i, j] * F.linear(hvs[i], hvs[j])
        if ((float(const + torch.var(hvs)) >= 0.001 or float(const + torch.var(hvs)) <= -0.001) and float(torch.norm(self.M))!=0):
            G_t = torch.matmul(self.covarianceMatrix[kwargs['ts']], hvs) / (const + torch.var(hvs))
            
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

    def train(self, sets_training, matrix_1_norm, y, epochs, sets_cv):

        for _ in range(epochs): # Number of iterations for all the samples
            
            for n in tqdm(range(matrix_1_norm.shape[0])):
                samples = matrix_1_norm[n, :]
            
                for i in (sets_training):
                
                    label = torch.tensor(samples[i+self.size])
                    sample = samples[i:i+self.size]

                    self.model_update(sample, label, ts = n, time = i) # Pass input and label to train
                    predictions_testing, enc, hvs = self(sample, ts = n)
                    y[n, i+self.size] = float(predictions_testing)
                
                if (n % self.opt.print_freq == 0):
                    pred, labels_full = self.test(sets_cv, matrix_1_norm, y)
                    print(f"Cross Validation root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
    
    def test(self, sets_testing, matrix_1_norm, y, cv = True):
        pred = []
        labels_full = []
        for i in (sets_testing):
            samples = matrix_1_norm[:, i:i+self.size]
            labels = matrix_1_norm[:, i+self.size]
            for n in range(samples.shape[0]):
                label = torch.tensor(labels[n])
                sample = samples[n, :]
                # Pass samples from test to model (forward function)
                predictions, enc, hvs = self(sample, ts = n)
                pred.append(float(predictions))
                y[n, i+self.size] = float(predictions)
                labels_full.append(float(label.unsqueeze(dim=0)))
        if (not cv):
            print(
            f"Testing root mean squared error of testing {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
        
        return pred, labels_full


def Return_Model(size, d, models, number_ts, opt):

    model_hd = RegHD_AR(size, d, models, number_ts, opt = opt)  # 1 class, 5

    return model_hd