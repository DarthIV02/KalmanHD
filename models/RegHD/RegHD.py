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
    def __init__(self, size, d, models, number_ts, **kwargs):
        super(RegHD, self).__init__()

        self.size = size
        self.d = d
        self.lr = kwargs['opt'].learning_rate # alpha
        self.M = torch.zeros(models, d).double() # Model initializes in 0
        self.opt = kwargs['opt']
        if self.opt.add_weights == 'false':
            self.project = embeddings.Projection(self.size, d).double() # 5 features, 10000 dimensions = hypervectors like weights?
        else:
            self.project = embeddings.Projection(1, d).double() # 5 features, 10000 dimensions = hypervectors like weights?
        self.project.weight.data.normal_(0, 1) # Normal distributions mean=0.0, std=1.0
        self.bias = nn.parameter.Parameter(torch.empty(d), requires_grad=False)
        self.bias.data.uniform_(0, 2 * math.pi) # bias
        self.cluster = functional.random_hv(models, d) 

    def encode(self, x): # encoding a single value TENSOR
        if self.opt.add_weights == 'false':
            for i in range(len(x)):
                x[i] = float(x[i])
        enc = self.project(x)
        sample_hv = torch.cos(enc + self.bias) * torch.sin(enc) 
        return functional.hard_quantize(sample_hv)
    
    def model_update(self, x, y, **kwargs): # update # y = no hv
        model_result, enc = self(x, ts = kwargs['ts'])
        update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
        self.M = update # New 
        # update cluster center?
        confidence = np.transpose(softmax(cos_similarity(self.cluster, enc)))
        center = [num.item() for num in confidence[0]].index(max(confidence[0]).item())
        self.cluster[center] = self.cluster[center] + (1-max(confidence[0])) * enc
        return center
    
    def forward(self, x, **kwargs): # With weights x: array of values
        if self.opt.add_weights != 'false':
            enc = torch.empty((0, self.d), dtype=torch.float32)
            for i, v in enumerate(x):
                hv = torch.reshape(self.encode(torch.tensor([v])), (1, self.d)) * self.alpha[kwargs['ts']][i]
                enc = torch.cat((enc, hv), 0)
            enc = functional.hard_quantize(multiset(enc))
        else:
            x = torch.tensor(x)
            enc = torch.reshape(self.encode(x), (1, self.d))
        confidence = np.transpose(softmax(cos_similarity(self.cluster, enc))) # Compare input with cluster
        model_result = F.linear(enc.type(torch.FloatTensor), self.M.type(torch.FloatTensor))
        res = F.linear(confidence, model_result) # Multiply enc (x) * weights (Model) = Dot product
        return res[0].clone().detach(), enc # Return the resolutions

    def train(self, sets_training, matrix_1_norm, epochs):
        for _ in range(epochs): # Number of iterations for all the samples
            pred = []
            labels_full = []
            for i in tqdm(sets_training):
                samples = matrix_1_norm[:, i:i+self.size]
                labels = matrix_1_norm[:, i+self.size]
                
                for n in range(samples.shape[0]):
                    label = torch.tensor(labels[n])
                    sample = samples[n, :]
                    if self.opt.add_weights == 'false' or self.opt.add_weights == 'Kalman Filter':
                        self.model_update(sample, label, ts = n) # Pass input and label to train
                    elif self.opt.add_weights == 'Yule Walker':
                        self.model_update(sample, label, ts = n, samples_until_pred = matrix_1_norm[n, :i+self.size+1])
                    predictions_testing, enc = self(sample, ts = n) # Pass samples from test to model (forward function)
                    pred.append(float(predictions_testing))
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
                sample = samples[n, :]
                # Pass samples from test to model (forward function)
                predictions, enc = self(sample, ts = n)
                pred.append(float(predictions))
                labels_full.append(float(label.unsqueeze(dim=0)))

        print(
            f"Testing root mean squared error of testing {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
        
        return pred, labels_full
    
    # This is to do the testing without knowing the true values after training
    """ def test2(self, start_testing, matrix_1_norm, number_predictions):
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
        
        return y, labels_full """

def bind_timeseries(self, x): # encoding self.size values
    for i in range(len(x)):
        x[i] = float(x[i])
    sample_hv = bind(self.position.weight, self.value(x))
    sample_hv = multiset(sample_hv)
    return hard_quantize(sample_hv)

def time_encoding(self, x): # encoding self.size values
    for i in range(len(x)):
        x[i] = float(x[i])
    sample_hv = bind(self.position.weight, self.value(x))
    for i, hv in enumerate(sample_hv):
        sample_hv[i] = permute(hv, shifts=i)
    sample_hv = multiset(sample_hv)
    return hard_quantize(sample_hv)

def linear_encoding(self, x):
    try:
        for i in range(len(x)):
            x[i] = float(x[i])
    except:
        pass
    sample_hv = bind(self.position.weight, self.value(x))
    sample_hv = multiset(sample_hv)
    return hard_quantize(sample_hv)

def yule_walker_update(self, x, y, **kwargs):
    model_result, enc = self(x, ts = kwargs['ts'])
    if (np.isnan(y)):
        y = model_result
    matrix = kwargs['samples_until_pred'] # Pass specific time serie of all the past samples
    for i, v in enumerate(matrix):
        hv = torch.reshape(self.encode(torch.tensor([v])), (1, self.d))
        matrix[i] = F.linear(hv.type(torch.FloatTensor), self.M.type(torch.FloatTensor))
    t = matrix.shape[0]
    mu_t = np.sum(matrix)/(t)
    dif = np.subtract(matrix, mu_t)
    sigma_t = np.sum(np.transpose(dif)**2)/(t-1)
    gama = np.empty((self.size+1))
    gama[0] = 1
    for k in range(1, self.size+1):
        sum = 0
        for q in range(t-self.size):
            sum += (matrix[q]-mu_t)*(matrix[q+k]-mu_t)
        if (sigma_t == 0):
            gama[k] = 0
        else:
            gama[k] = sum/(sigma_t)*(t-k)            
    R = np.empty((self.size,self.size))
    for i in range(self.size):
        for j in range(self.size):
            R[i,j] = gama[j-i]
            R[j,i] = gama[j-i]
    gama = np.transpose(gama)
    self.alpha[kwargs['ts']] = np.dot(np.linalg.inv(R), gama[1:])
    update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
    self.M = update # New 
    confidence = np.transpose(softmax(cos_similarity(self.cluster, enc)))
    center = [num.item() for num in confidence[0]].index(max(confidence[0]).item())
    self.cluster[center] = self.cluster[center] + (1-max(confidence[0])) * enc
    return center

def KalmanFilterUpdate(self, x, y, **kwargs):
    model_result, enc = self(x, ts = kwargs['ts'])
    if (np.isnan(y)):
        y = model_result
    self.mse.update(model_result, y)
    sigma_2 = self.mse.compute().item()
    enc_hv = torch.empty((0, self.d), dtype=torch.float32)
    for v in x:
        hv = torch.reshape(self.encode(torch.tensor([v])), (1, self.d))
        enc_hv = torch.cat((enc_hv, hv), 0)
    H_t = np.dot(self.M, np.transpose(enc_hv))
    G_t = (np.dot(self.covarianceMatrix[kwargs['ts']], np.transpose(H_t))) / (np.dot(np.dot(H_t, self.covarianceMatrix[kwargs['ts']]), np.transpose(H_t)) + sigma_2)
    G_t2 = torch.empty(20)
    for i in range(len(G_t)):
        G_t2[i] = G_t[i][0]
    A_t = y - model_result
    self.alpha[kwargs['ts']] = np.add(self.alpha[kwargs['ts']], G_t2*float(A_t))
    self.covarianceMatrix[kwargs['ts']] = np.subtract(self.covarianceMatrix[kwargs['ts']], G_t2*H_t*self.covarianceMatrix[kwargs['ts']])
    update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
    self.M = update # New 
    confidence = np.transpose(softmax(cos_similarity(self.cluster, enc)))
    center = [num.item() for num in confidence[0]].index(max(confidence[0]).item())
    self.cluster[center] = self.cluster[center] + (1-max(confidence[0])) * enc
    return center

class RegHD_Kmeans(RegHD):
    def __init__(self, size, d, models, number_ts, **kwargs):
        super().__init__(size, d, models, number_ts, **kwargs)   
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
    def __init__(self, size, d, models, number_ts, **kwargs):
        super().__init__(size, d, models, number_ts, **kwargs)
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

def Return_Model(size, d, models, number_ts, opt):

    if(opt.clustering == "none"):
        model_hd = RegHD(size, d, models, number_ts, opt = opt)  # 1 class, 5
    elif(opt.clustering == "kmeans"):
        model_hd = RegHD_Kmeans(size, d, models, number_ts, opt = opt)
    elif(opt.clustering == "spectral_clustering"):
        model_hd = RegHD_SpectralClustering(size, d, models, number_ts, opt = opt)

    model_hd.lr = opt.learning_rate

    if(opt.hd_encoder != "nonlinear"):
        model_hd.position = embeddings.Random(size, d)
        model_hd.value = embeddings.Level(opt.levels, d)
        if (opt.hd_encoder == "bind_timeseries"):
            model_hd.encode = types.MethodType(bind_timeseries, model_hd)
        elif(opt.hd_encoder == "time_encoding"):
            model_hd.encode = types.MethodType(time_encoding, model_hd)
        elif(opt.hd_encoder == "linear"):
            model_hd.encode = types.MethodType(linear_encoding, model_hd)
    
    if (opt.add_weights != "false"):
        model_hd.alpha = {}
        for i in range(number_ts):
            model_hd.alpha[i] = np.random.rand(size)
        if (opt.add_weights == "Yule Walker"):
            model_hd.model_update = types.MethodType(yule_walker_update, model_hd)
        if (opt.add_weights == "Kalman Filter"):
            model_hd.covarianceMatrix = {}
            for i in range(number_ts):
                model_hd.covarianceMatrix[i] = np.identity(size)*np.random.randint(100, 200)
            model_hd.mse = torchmetrics.MeanSquaredError()
            model_hd.model_update = types.MethodType(KalmanFilterUpdate, model_hd)

    return model_hd
