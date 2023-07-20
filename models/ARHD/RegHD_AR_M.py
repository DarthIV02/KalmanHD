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
        self.opt = kwargs['opt']
        self.project = Projection_2(self.size, d, dtype=torch.float32) # 5 features, 10000 dimensions = hypervectors like weights?
        self.project.weight.data.normal_(0, 1) # Normal distributions mean=0.0, std=1.0
        self.bias = nn.parameter.Parameter(torch.empty(d, self.size), requires_grad=False)
        self.bias.data.uniform_(0, 2 * math.pi) # bias
        self.kwargs = kwargs
        """self.covarianceMatrix = {}
        for i in range(number_ts):
            self.covarianceMatrix[i] = 0.1*torch.eye(self.d) # Process noise covariance matrix"""
        self.covarianceMatrix = 0.1*torch.eye(self.d)
        """self.alpha = {}
        for i in range(number_ts):
            self.alpha[i] = torch.zeros(self.d, 1) # Initial state estimate"""
        self.alpha = [torch.zeros(d, 1).float()]
        self.var = [0]
        self.current_ts = None
        self.cluster = []

    def hard_quantize(self, hv):
        hv = (hv+self.size)/((self.size)*(2**(1-self.opt.hd_representation)))
        hv = torch.floor(hv)
        return hv
    
    def flip_bits(self, enc):
        total_bits = self.d * self.opt.hd_representation
        flip_positions = np.random.choice(total_bits, int(self.opt.flipping_rate * total_bits), replace=False)
        for pos in flip_positions:
            value = list(format(int(enc[pos//self.opt.hd_representation]), 'b').rjust(self.opt.hd_representation, '0'))
            if(value[pos % self.opt.hd_representation] == '0'):
                value[pos % self.opt.hd_representation] = '1'
            else:
                value[pos % self.opt.hd_representation] = '0'
            value = "".join(value)
            enc[pos//self.opt.hd_representation] = int(value,2)
        return enc
    
    def encode(self, x, **kwargs): # encoding a single value TENSOR
        #x_2 = x * self.alpha[kwargs['ts']]
        enc = self.project(torch.reshape(x, (1, self.size)))
        enc = torch.cos(enc + self.bias) * torch.sin(enc) 
        # return self.hard_quantize(multiset(torch.transpose(enc, 0, 1))) <-- Original with RegHD
        enc = self.hard_quantize(multiset(torch.transpose(enc, 0, 1)))
        if self.opt.flipping_rate > 0:
            enc = self.flip_bits(enc)
        return enc


    def model_update(self, x, y, **kwargs): # update # y = no hv

        if(self.current_ts != kwargs['ts']):
            self.current_ts = kwargs['ts']
            self.covarianceMatrix = 0.1*torch.eye(self.d)

        model_result, enc, index, novel = self(x, ts = kwargs['ts'])
        enc = torch.reshape(enc, (1, self.d))

        if len(self.cluster) == 0:
            self.cluster.append(enc)
        else:
            if novel and len(self.cluster) < self.opt.models:
                index = len(self.cluster)
                self.cluster.append(enc)
                self.alpha.append(torch.zeros(self.d, 1).float())
                self.var.append(0)
            else:
                self.cluster[index] = bundle(self.cluster[index], enc)

        x = torch.reshape(torch.tensor(x, dtype = torch.float32), (1, self.size))
        const = torch.matmul(self.covarianceMatrix, torch.transpose(enc, 0, 1))
        complete = torch.matmul(enc,const)
        #const = torch.matmul(enc, torch.matmul(self.covarianceMatrix, torch.transpose(enc, 0, 1)))
        self.var[index] = (self.opt.alpha * self.var[index]) + (1-self.opt.alpha) * torch.var(x)
        #self.alpha[kwargs['ts']] += torch.transpose(float(self.lr) * A_t * enc, 0, 1)
        A_t = float(y - model_result)
        if (float(complete + self.var[index]) >= 0.001 or float(complete + self.var[index]) <= -0.001):
        #if(torch.var(x) >= 0.0001 or torch.var(x) <= -0.0001):
            G_t = const / (complete + self.var[index])
    
            
        #for i in range(kwargs['ts'], len(self.alpha)):
            self.alpha[index] += G_t*A_t*self.opt.learning_rate
            #self.alpha += (torch.transpose(enc, 0, 1) / (const + torch.var(x)))*A_t*self.lr
            #self.alpha[kwargs['ts']] += (torch.transpose(enc, 0, 1)/self.var)*A_t
            #self.alpha[kwargs['ts']] += torch.transpose(float(self.lr) * A_t * enc, 0, 1)
            #self.alpha += torch.transpose(float(self.lr) * A_t * ((enc) / self.var), 0, 1)
            #self.covarianceMatrix -= torch.matmul(torch.matmul(G_t, enc), self.covarianceMatrix)
            self.covarianceMatrix -= torch.matmul(G_t, torch.transpose(const, 0, 1))
    
    def forward(self, x, **kwargs): # With weights x: array of values
        x = torch.tensor(x.reshape((self.size, 1)), dtype = torch.float32)
        
        enc = self.encode(x, ts = kwargs['ts'])

        try:
            sim = [cos_similarity(enc, self.cluster[i]) for i in range(len(self.cluster))]
            novel = all(float(s) < 1-self.opt.novelty for s in sim)
            index = sim.index(max(sim))
        except:
            index = 0

        
        
        #enc = self.encode(x, ts = kwargs['ts'])
        #enc = torch.reshape(enc, (1, self.d))
        model_result = F.linear(enc.type(torch.FloatTensor), torch.transpose(self.alpha[index].type(torch.FloatTensor), 0, 1))
        #model_result = torch.sum(enc)
        #self.alpha[kwargs['ts']] = torch.reshape(self.alpha[kwargs['ts']], (self.size, self.d))
        
        return model_result, enc, index, novel

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
                    
                    predictions_testing, enc, index, novel = self(sample, ts = n)
                    y[n, i+self.size] = float(predictions_testing)
                
                if (n % self.opt.print_freq == 0):
                    pred, labels_full = self.test(sets_cv, matrix_1_norm, matrix_1_norm_org, y)
                    print(f"\nCross Validation root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
                    #print(f"Self.var = {self.var}")
    
    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv = True):
        pred = []
        labels_full = []
        for i in (sets_testing):
            samples = matrix_1_norm_org[:, i:i+self.size]
            labels = matrix_1_norm[:, i+self.size]
            for n in range(samples.shape[0]):
                sample = samples[n, :]
                if(np.isnan(labels[n])):
                    predictions, enc = self(sample, ts = n)
                    matrix_1_norm[n, i+self.size] = predictions
                label = torch.tensor(labels[n])
                # Pass samples from test to model (forward function)
                predictions, enc, index, novel= self(sample, ts = n)
                pred.append(float(predictions))
                y[n, i+self.size] = float(predictions)
                labels_full.append(matrix_1_norm_org[n, i+self.size])
        if (not cv):
            error = mean_squared_error(labels_full, pred, squared=False)
            print(
            f"Testing root mean squared error of testing {(error):.3f}")
            return error
        
        return pred, labels_full
    
    def test2(self, start_testing, matrix_1_norm, number_predictions):
        y = np.zeros((matrix_1_norm.shape[0], number_predictions+self.size))
        labels_full = matrix_1_norm[:, start_testing+self.size:]
        y[:, :self.size] = matrix_1_norm[:, start_testing:start_testing+self.size]
        for i in tqdm(range(number_predictions)):
            samples = y[:, i:i+self.size]
            pred = []
            for n in range(samples.shape[0]):
                sample = torch.tensor(samples[n, :])
                # Pass samples from test to model (forward function)
                predictions = self.forward(sample)
                pred.append(float(predictions[0]))
            y[:, self.size+i] = pred

        print(
            f"Testing root mean squared error of testing {(mean_squared_error(labels_full.flatten(), y[:, -number_predictions:].flatten(), squared=False)):.3f}")
        
        return y, labels_full


def Return_Model(size, d, models, number_ts, opt):

    torch.random.manual_seed(opt.trial)

    model_hd = RegHD_AR(size, d, models, number_ts, opt = opt)  # 1 class, 5

    return model_hd
