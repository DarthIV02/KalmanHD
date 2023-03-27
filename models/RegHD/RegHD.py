# For RegHD
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchmetrics
from tqdm import tqdm

from torchhd import functional
from torchhd import embeddings

import numpy as np
# Model based on RegHD application for Single model regression -> No comparing which cluster


class SingleModel_With_NonLinear_Sin(nn.Module):
    def __init__(self, num_classes, size, d):
        super(SingleModel_With_NonLinear_Sin, self).__init__()

        self.lr = 0.00001  # alpha
        self.M = torch.zeros(1, d).double()  # Model initializes in 0
        # 5 features, 10000 dimensions = hypervectors like weights?
        self.project = embeddings.Projection(size, d).double()
        # Normal distributions mean=0.0, std=1.0
        self.project.weight.data.normal_(0, 1)
        self.bias = nn.parameter.Parameter(torch.empty(d), requires_grad=False)
        self.bias.data.uniform_(0, 2 * math.pi)  # bias

    def encode(self, x):  # encoding a value
        for i in range(len(x)):
            x[i] = float(x[i])
        enc = self.project(x)
        sample_hv = torch.cos(enc + self.bias) * torch.sin(enc)
        return functional.hard_quantize(sample_hv)

    def model_update(self, x, y):  # update # y = no hv
        # Model + alpha*(Error)*(x)
        update = self.M + self.lr * (y - (F.linear(x, self.M))) * x
        # update = update.mean(0) # Mean by columns
        self.M = update  # New

    def forward(self, x):
        enc = self.encode(x)
        # Multiply enc (x) * weights (Model) = Dot product
        res = F.linear(enc, self.M)
        return res  # Return the resolutions


def Return_Model():
    model_hd = SingleModel_With_NonLinear_Sin(2, 40, 10000)  # 1 class, 5
    return model_hd


def Train_Model(model_hd, matrix, sets_training):
    time_arr = []
    # main()
    # print("--- %s seconds ---" % (time.time() - start_time))

    with torch.no_grad():  # disabled gradient calculation because were doing it manually
        for _ in tqdm(range(10)):  # Number of iterations for all the samples
            mse = torchmetrics.MeanSquaredError()
            for i in sets_training:  # Calculated at random
                pred = []
                samples = matrix[:, i:i+40]
                labels = matrix[:, i+40]
                # print(f"sample: {samples}, label:{labels}")
                # samples = samples.to(device) # pass sample and label (1 at a time)
                # labels = labels.to(device)

                for n in range(samples.shape[0]):
                    label = torch.tensor(labels[n])
                    sample = torch.tensor(samples[n, :])
                    # samples =
                    samples_hv = model_hd.encode(sample)  # Encode the inputs
                    # print(f"sample_hv:{samples_hv}")
                    # Pass input and label to train
                    model_hd.model_update(samples_hv, label)

                    # Pass samples from test to model (forward function)
                    predictions_testing = model_hd(sample)
                    # print(predictions_testing.item())
                    pred.append(predictions_testing)
                    # predictions_testing = predictions_testing
                    # label = label
                    mse.update(predictions_testing, label.unsqueeze(dim=0))
                    # mse.update(predictions_testing.cpu(), label)

                    # time_arr.append((time.time() - start_time, np.absolute(np.sum(labels-pred))))
                    time_arr.append(np.absolute(
                        label.item()-predictions_testing.item()))

            print(
                f"Training mean squared error of {(mse.compute().item()):.3f}")

    return model_hd, time_arr


def Test_Model(model_hd, matrix, sets_testing):
    pred_hd = []
    original_hd = []

    with torch.no_grad():
        mse = torchmetrics.MeanSquaredError()
        for i in tqdm(sets_testing):
            pred = []
            samples = matrix[:, i:i+40]
            labels = matrix[:, i+40]
            for n in range(samples.shape[0]):
                label = torch.tensor(labels[n])
                sample = torch.tensor(samples[n, :])
                # Pass samples from test to model (forward function)
                predictions = model_hd(sample)
                pred.append(predictions)
                # predictions = predictions * TARGET_STD + TARGET_MEAN # What is target
                # labels = labels * TARGET_STD + TARGET_MEAN
                mse.update(predictions, label.unsqueeze(dim=0))
                # dif_hd.append(np.absolute(label-predictions))
                pred_hd.append(predictions)
                original_hd.append(label)

        print(
            f"Testing mean squared error of testing {(mse.compute().item()):.3f}")

    return model_hd, pred_hd, original_hd
