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

from torchhd import functional, embeddings, bind, multiset, hard_quantize, permute

from scipy.special import softmax

from sklearn.cluster import KMeans, SpectralClustering

import time

# Model based on RegHD application for Single model regression -> No comparing which cluster
class RegHD(nn.Module):
    def __init__(self, size, d, models, number_ts, **kwargs):
        super(RegHD, self).__init__()

        self.size = size
        self.d = d
        self.dev = torch.device(kwargs['dev'])
        self.lr = torch.tensor([kwargs['opt'].learning_rate]).to(self.dev) # alpha
        #self.M = torch.zeros(models, d).float() # Model initializes in 0
        self.M = torch.zeros(1, d).float().to(self.dev)
        self.opt = kwargs['opt']
        self.project = embeddings.Projection(self.size, d, dtype=torch.float32, device=self.dev) # 5 features, 10000 dimensions = hypervectors like weights?
        #self.project = embeddings.Projection(1, d).float()
        self.project.weight.data.normal_(0, 1) # Normal distributions mean=0.0, std=1.0
        self.bias = nn.parameter.Parameter(torch.empty(d), requires_grad=False).to(self.dev)
        self.bias.data.uniform_(0, 2 * math.pi) # bias
        #self.cluster = functional.random_hv(models, d) 
        self.kwargs = kwargs

    def hard_quantize(self, hv):
        hv = (hv+self.size)/((self.size)*(2**(1-self.opt.hd_representation)))
        hv = torch.floor(hv)
        return hv
    
    """def flip_bits(self, enc):
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
        return enc"""

    def encode(self, x, **kwargs): # encoding a single value TENSOR
        enc = self.project(torch.reshape(x, (1, self.size)))
        enc = torch.cos(enc + self.bias) * torch.sin(enc) 
        #return functional.hard_quantize(enc)
        #return self.hard_quantize(enc) <-- Original
        enc = hard_quantize(enc)
        #if self.opt.flipping_rate > 0:
        #    enc = self.flip_bits(enc[0])
        return enc
    
    def model_update(self, x, y, **kwargs): # update # y = no hv
        model_result, enc = self(x)
        #print(self.M.device)
        #print(self.lr.device)
        #print(y.device)
        #print(model_result.device)
        #print(enc.device)
        self.M += (self.lr * (y - model_result) * enc)
        #update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
        #self.M = update # New 
        # update cluster center?
        #confidence = np.transpose(softmax(cos_similarity(self.cluster, enc)))
        #center = [num.item() for num in confidence[0]].index(max(confidence[0]).item())
        #self.cluster[center] = self.cluster[center] + (1-max(confidence[0])) * enc
        #return center
    
    def forward(self, x, **kwargs): # With weights x: array of values
        #x = torch.tensor(x * self.alpha[kwargs['ts']])
        #x = torch.tensor(x.reshape((self.size, 1)), dtype = torch.float32, device=self.dev)
        enc = self.encode(x)
        enc = torch.reshape(enc, (1, self.d))

        model_result = torch.sum(torch.mul(enc, self.M))

        #model_result = F.linear(enc.type(torch.FloatTensor), self.M.type(torch.FloatTensor))
        #res = F.linear(confidence, model_result) # Multiply enc (x) * weights (Model) = Dot product
        #return res[0].clone().detach(), enc, hvs # Return the resolutions
        return model_result, enc

    def train(self, sets_training, matrix_1_norm, matrix_1_norm_org, y, epochs, sets_cv):

        size = matrix_1_norm.shape[0]
        matrix_1_norm = torch.tensor(matrix_1_norm, dtype = torch.float32, device=self.dev)
        y = torch.tensor(y, dtype = torch.float32, device=self.dev)

        for _ in range(epochs): # Number of iterations for all the samples
            
            for n in tqdm(range(size)):
                samples = matrix_1_norm[n, :]
                #samples = torch.tensor(matrix_1_norm[n, :], dtype = torch.float32, device=self.dev)
            
                for i in (sets_training):
                    
                    sample = samples[i:i+self.size]
                    #sample = torch.tensor(sample.reshape((self.size, 1)), dtype = torch.float32, device=self.dev)
                    #label = torch.tensor(samples[i+self.size]).to(self.dev)
                    label = samples[i+self.size]
                    self.model_update(sample, label, ts = n) # Pass input and label to train
                    predictions_testing, enc = self(sample, ts = n)
                    #pred.append(float(predictions_testing))
                    y[n, i+self.size] = predictions_testing
                    #labels_full.append(float(label.unsqueeze(dim=0)))
                
                if (n % self.opt.print_freq == 0 and n != 0):
                    pred, labels_full = self.test(sets_cv, matrix_1_norm, matrix_1_norm_org, y.cpu())
                    print(f"Cross Validation root mean squared error of {(mean_squared_error(labels_full, pred, squared=False)):.3f}")
    
    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv = True):
        matrix_1_norm_org = torch.tensor(matrix_1_norm_org, dtype = torch.float32, device=self.dev)
        matrix_1_norm = torch.tensor(matrix_1_norm, dtype = torch.float32, device=self.dev)
        pred = []
        labels_full = []
        for i in (sets_testing):
            samples = matrix_1_norm[:, i:i+self.size]
            #labels = matrix_1_norm[:, i+self.size]
            for n in range(samples.shape[0]):
                sample = samples[n, :]
                #sample = torch.tensor(sample.reshape((self.size, 1)), dtype = torch.float32, device=self.dev)
                """if(np.isnan(labels[n])):
                    predictions, enc = self(sample, ts = n)
                    matrix_1_norm[n, i+self.size] = predictions"""
                #label = torch.tensor(labels[n])
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

"""def yule_walker_update(self, x, y, **kwargs):
    model_result, enc, hvs = self(x, ts = kwargs['ts'])
    if (np.isnan(y)):
        y = model_result
    matrix = np.copy(kwargs['samples_until_pred']) # Pass specific time serie of all the past samples
    enc = self.project(torch.tensor(matrix.reshape(matrix.shape[0], 1), dtype=torch.float32))
    hv = torch.reshape(torch.cos(enc + self.bias) * torch.sin(enc), (1, self.d))
    matrix[i] = F.linear(hv.type(torch.FloatTensor), self.M.type(torch.FloatTensor))
    t = matrix.shape[0]
    mu_t = np.sum(matrix)/(t)
    dif = np.subtract(matrix, mu_t)
    sigma_t = np.sum(np.transpose(dif)**2)/(t-1)
    gama = np.empty((self.size+1))
    gama[0] = 1
    for k in range(1, self.size+1):
        sum = 0
        for q in range(t-k):
            sum += (matrix[q]-mu_t)*(matrix[q+k]-mu_t)
        if (sigma_t == 0):
            gama[k] = 0
        else:
            gama[k] = sum/((sigma_t)*(t-k))            
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
    return center"""

"""def KalmanFilterUpdate(self, x, y, **kwargs): # First test
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
    G_t2 = torch.empty(self.size)
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
    return center"""

"""def KalmanFilterUpdate2(self, x, y, **kwargs): # If values (original) are multiplied by alpha
    model_result, enc = self(x, ts = kwargs['ts'])
    if (np.isnan(y)):
        y = model_result
    self.mse.update(torch.tensor(float(model_result)), y)
    sigma_2 = self.mse.compute().item()
    H_t = x
    G_t = (np.dot(self.covarianceMatrix[kwargs['ts']], np.transpose(H_t))) / (np.dot(np.dot(H_t, self.covarianceMatrix[kwargs['ts']]), np.transpose(H_t)) + sigma_2)
    A_t = y - model_result
    self.alpha[kwargs['ts']] = np.add(self.alpha[kwargs['ts']], G_t*float(A_t))
    self.covarianceMatrix[kwargs['ts']] = np.subtract(self.covarianceMatrix[kwargs['ts']], G_t*H_t*self.covarianceMatrix[kwargs['ts']])
    update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
    self.M = update # New 
    confidence = np.transpose(softmax(cos_similarity(self.cluster, enc)))
    center = [num.item() for num in confidence[0]].index(max(confidence[0]).item())
    self.cluster[center] = self.cluster[center] + (1-max(confidence[0])) * enc
    return center"""

"""def KalmanFilterUpdate3(self, x, y, **kwargs): # Where alpha is multiplied by hypervectors in a single pass
    model_result, enc, hvs = self(x, ts = kwargs['ts'])
    enc = enc[0]
    if (np.isnan(y)):
        y = model_result
    self.mse.update(torch.tensor(float(model_result)), y)
    sigma_2 = self.mse.compute().item()
    H_t = np.dot(self.M, torch.transpose(hvs, 0, 1))[0]
    G_t = (np.dot(self.covarianceMatrix[kwargs['ts']], np.transpose(H_t))) / (np.dot(np.dot(H_t, self.covarianceMatrix[kwargs['ts']]), np.transpose(H_t)) + sigma_2)
    A_t = y - model_result
    self.alpha[kwargs['ts']] = np.add(self.alpha[kwargs['ts']], torch.tensor(G_t*float(A_t)))
    self.covarianceMatrix[kwargs['ts']] = np.subtract(self.covarianceMatrix[kwargs['ts']], G_t*H_t*self.covarianceMatrix[kwargs['ts']])
    update = self.M + (float(self.lr) * float(y - model_result) * enc) # Model + alpha*(Error)*(x)
    self.M = update # New 
    confidence = np.transpose(softmax(cos_similarity(self.cluster, enc)))
    center = [num.item() for num in confidence[0]].index(max(confidence[0]).item())
    self.cluster[center] = self.cluster[center] + (1-max(confidence[0])) * enc
    return center"""

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

"""class RegHD_SpectralClustering (RegHD):
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
                    self.model_update(encoded[item], label, self.cluster_model[item]) # Pass input and label to train"""

def Return_Model(size, d, models, number_ts, opt, dev):

    """if(opt.clustering == "none"):
        model_hd = RegHD(size, d, models, number_ts, opt = opt)  # 1 class, 5
    elif(opt.clustering == "kmeans"):
        model_hd = RegHD_Kmeans(size, d, models, number_ts, opt = opt)
    elif(opt.clustering == "spectral_clustering"):
        model_hd = RegHD_SpectralClustering(size, d, models, number_ts, opt = opt)"""

    model_hd = RegHD(size, d, models, number_ts, opt = opt, dev = dev)
    model_hd.to(dev)

    #model_hd.lr = opt.learning_rate

    if(opt.hd_encoder != "nonlinear"):
        model_hd.position = embeddings.Random(size, d)
        model_hd.value = embeddings.Level(opt.levels, d)
        if (opt.hd_encoder == "bind_timeseries"):
            model_hd.encode = types.MethodType(bind_timeseries, model_hd)
        elif(opt.hd_encoder == "time_encoding"):
            model_hd.encode = types.MethodType(time_encoding, model_hd)
        elif(opt.hd_encoder == "linear"):
            model_hd.encode = types.MethodType(linear_encoding, model_hd)
    
    """if (opt.add_weights != "false"):
        model_hd.alpha = {}
        for i in range(number_ts):
            #model_hd.alpha[i] = np.random.rand(size)
            model_hd.alpha[i] = torch.rand(size)
        if (opt.add_weights == "Yule Walker"):
            model_hd.model_update = types.MethodType(yule_walker_update, model_hd)
        if (opt.add_weights == "Kalman Filter"):
            model_hd.covarianceMatrix = {}
            for i in range(number_ts):
                model_hd.covarianceMatrix[i] = np.identity(size)*np.random.randint(100, 200)
            model_hd.mse = torchmetrics.MeanSquaredError()
            model_hd.model_update = types.MethodType(KalmanFilterUpdate3, model_hd)"""

    return model_hd
