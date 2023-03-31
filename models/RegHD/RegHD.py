# For RegHD
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchmetrics
from tqdm import tqdm

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

from torchhd import functional, embeddings, cos_similarity, bind, multiset, hard_quantize, permute

from scipy.special import softmax

from sklearn.cluster import KMeans, SpectralClustering

import time

# Model based on RegHD application for Single model regression -> No comparing which cluster
class RegHD(nn.Module):
    def __init__(self, size, d, models, **kwargs):
        super(RegHD, self).__init__()

        self.lr = kwargs['opt'].learning_rate # alpha
        self.M = torch.zeros(models, d).double() # Model initializes in 0
        self.project = embeddings.Projection(size, d).double() # 5 features, 10000 dimensions = hypervectors like weights?
        self.project.weight.data.normal_(0, 1) # Normal distributions mean=0.0, std=1.0
        self.bias = nn.parameter.Parameter(torch.empty(d), requires_grad=False)
        self.bias.data.uniform_(0, 2 * math.pi) # bias
        self.cluster = functional.random_hv(models, d)
        self.size = size
        self.d = d
        self.opt = kwargs['opt']

    def encode(self, x): # encoding a value
        for i in range(len(x)):
            x[i] = float(x[i])
        enc = self.project(x)
        sample_hv = torch.cos(enc + self.bias) * torch.sin(enc) 
        return functional.hard_quantize(sample_hv)

    def model_update(self, x, y): # update # y = no hv
        x = torch.reshape(x, (1,self.d))
        confidence = np.transpose(softmax(cos_similarity(self.cluster, x))) # Compare input with cluster
        model_result = F.linear(x.type(torch.FloatTensor), self.M.type(torch.FloatTensor))
        update = self.M + (float(self.lr) * float(y - F.linear(confidence, model_result)) * x) # Model + alpha*(Error)*(x)
        #update = update.mean(0) # Mean by columns
        self.M = update # New 
        # update cluster center?
        center = [num.item() for num in confidence[0]].index(max(confidence[0]).item())
        self.cluster[center] = self.cluster[center] + (1-max(confidence[0])) * x
        return center
        

    def forward(self, x):
        enc = torch.reshape(self.encode(x), (1, self.d))
        confidence = np.transpose(softmax(cos_similarity(self.cluster, enc))) # Compare input with cluster
        model_result = F.linear(enc, self.M.type(torch.FloatTensor))
        res = F.linear(confidence, model_result) # Multiply enc (x) * weights (Model) = Dot product
        return res # Return the resolutions

    def train(self, sets_training, matrix_1_norm, epochs):
        for _ in range(epochs): # Number of iterations for all the samples
            pred = []
            labels_full = []
            for i in tqdm(sets_training):
                samples = matrix_1_norm[:, i:i+self.size]
                labels = matrix_1_norm[:, i+self.size]
                
                for n in range(samples.shape[0]):
                    label = torch.tensor(labels[n])
                    sample = torch.tensor(samples[n, :])
                    samples_hv = self.encode(sample) # Encode the inputs
                    self.model_update(samples_hv, label) # Pass input and label to train
                    predictions_testing = self.forward(sample) # Pass samples from test to model (forward function)
                    pred.append(float(predictions_testing[0]))
                    labels_full.append(float(label.unsqueeze(dim=0)))

            print(f"Training root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
    
    def test(self, sets_testing, matrix_1_norm):
        pred = []
        labels_full = []
        for i in tqdm(sets_testing):
            samples = matrix_1_norm[:, i:i+self.size]
            labels = matrix_1_norm[:, i+self.size]
            for n in range(samples.shape[0]):
                label = torch.tensor(labels[n])
                sample = torch.tensor(samples[n, :])
                # Pass samples from test to model (forward function)
                predictions = self.forward(sample)
                pred.append(float(predictions[0]))
                labels_full.append(float(label.unsqueeze(dim=0)))

        print(
            f"Testing root mean squared error of testing {(mean_squared_error(labels_full, pred, squared=False)):.3f}")

def bind_timeseries(self, x): # encoding a value
        for i in range(len(x)):
            x[i] = float(x[i])
        sample_hv = bind(self.position.weight, self.value(x))
        sample_hv = multiset(sample_hv)
        return hard_quantize(sample_hv)

def time_encoding(self, x): # encoding a value
        for i in range(len(x)):
            x[i] = float(x[i])
        sample_hv = bind(self.position.weight, self.value(x))
        for i, hv in enumerate(sample_hv):
            sample_hv[i] = permute(hv, shifts=i)
        sample_hv = multiset(sample_hv)
        return hard_quantize(sample_hv)

class RegHD_Kmeans(RegHD):
    def __init__(self, size, d, models, **kwargs):
        super().__init__(size, d, models, **kwargs)   
        self.cluster_model = None
        self.models = models

    def model_update(self, x, y, cluster): # update # y = no hv
        x = torch.reshape(x, (1,self.d))
        model_result = F.linear(x, self.M[cluster]).type(torch.FloatTensor)
        update = self.M[cluster] + (self.lr * (y - model_result) * x) # Model + alpha*(Error)*(x)
        self.M[cluster] = update[0] # New 
    
    def forward(self, x):
        enc = torch.reshape(self.encode(x), (1,self.d))
        model_result = F.linear(enc, self.M[self.cluster_model.predict(enc)])
        return model_result # Return the resolutions
    
    def train(self, sets_training, matrix_1_norm, epochs):
        
        full_samples = []
        for i in sets_training:
            samples = matrix_1_norm[:, i:i+self.size]
            for n in range(samples.shape[0]):
                sample = torch.tensor(samples[n, :])
                full_samples.append(sample)
        
        encoded = [self.encode(samp).numpy() for samp in full_samples]
        
        print("Kmeans Clusterization Started")
        
        if(self.cluster_model == None):
            kmeans = KMeans(n_clusters=self.models, random_state=0, n_init=10).fit(encoded)
            self.cluster_model = kmeans

        for _ in range(epochs): # Number of iterations for all the samples
            pred = []
            labels_full = []
            for time, i in tqdm(enumerate(sets_training)):
                labels = matrix_1_norm[:, i+self.size]
                
                for n in range(samples.shape[0]):
                    label = torch.tensor(labels[n])
                    item = (samples.shape[0]*time) + n
                    self.model_update(torch.tensor(encoded[item]), label, self.cluster_model.labels_[item]) # Pass input and label to train
                    predictions_testing = self(full_samples[item]) # Pass samples from test to model (forward function)
                    pred.append(float(predictions_testing[0]))
                    labels_full.append(float(label.unsqueeze(dim=0)))

            print(f"Training root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")

class RegHD_SpectralClustering (RegHD):
    def __init__(self, size, d, models, **kwargs):
        super().__init__(size, d, models, **kwargs)
        self.cluster_model = None
        self.training_samples = None
        self.models = models

    def model_update(self, x, y, cluster): # update # y = no hv
        x = torch.reshape(x, (1,self.d))
        model_result = F.linear(x.type(torch.FloatTensor), self.M[cluster].type(torch.FloatTensor))
        update = self.M[cluster] + (self.lr * (y - model_result) * x) # Model + alpha*(Error)*(x)
        self.M[cluster] = update[0] # New 

    def forward(self, x):
        enc = torch.reshape(self.encode(x), (1,self.d))
        similarity = cos_similarity(self.training_samples, enc)
        i = list(similarity.numpy()).index(max(similarity).numpy()[0])
        model_result = F.linear(enc, self.M[self.cluster_model[i]])
        return model_result # Return the resolutions
    
    def train(self, sets_training, matrix_1_norm, epochs):
        
        full_samples = {}
        for time, i in enumerate(sets_training):
            samples = matrix_1_norm[:, i:i+self.size]
            for n in range(0, samples.shape[0], 1):
                sample = torch.tensor(samples[n, :])
                full_samples[(samples.shape[0]*time) + n] = sample

        if(self.cluster_model == None):
            encoded = torch.zeros(len(full_samples), self.d)
            for i, samp in (full_samples.items()):
                encoded[i, :] = self.encode(samp)
            self.training_samples = encoded
            affinity_matrix = cos_similarity(encoded, encoded)
            clustering = SpectralClustering(n_clusters=self.models, assign_labels='discretize', 
                                            random_state=0, affinity='precomputed').fit(affinity_matrix)
            self.cluster_model = clustering.labels_

        for _ in range(epochs): # Number of iterations for all the samples
            for time, i in enumerate(sets_training):
                labels = matrix_1_norm[:, i+self.size]
                
                for n in range(samples.shape[0]):
                    label = torch.tensor(labels[n])
                    item = (samples.shape[0]*time) + n
                    self.model_update(encoded[item], label, self.cluster_model[item]) # Pass input and label to train

def Return_Model(size, d, models, opt):

    if(opt.clustering == "none"):
        model_hd = RegHD(size, d, models, opt = opt)  # 1 class, 5

    elif(opt.clustering == "kmeans"):
        model_hd = RegHD_Kmeans(size, d, models, opt = opt)

    elif(opt.clustering == "spectral_clustering"):
        model_hd = RegHD_SpectralClustering(size, d, models, opt = opt)

    model_hd.lr = opt.learning_rate

    if(opt.hd_encoder != "nonlinear"):
        model_hd.position = embeddings.Random(size, d)
        model_hd.value = embeddings.Level(opt.levels, d)
        if (opt.hd_encoder == "bind_timeseries"):
            model_hd.encode = types.MethodType(bind_timeseries, model_hd)
        elif(opt.hd_encoder == "time_encoding"):
            model_hd.encode = types.MethodType(time_encoding, model_hd)

    return model_hd
