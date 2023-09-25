# KalmanHD: Robust On-Device Time Series Forecasting with Hyperdimensional Computing

This repo contains the simulation implementation for paper:

Ivannia Gomez, Xiaofan Yu, Tajana Rosing. "KalmanHD: Robust On-Device Time Series Forecasting with Hyperdimensional Computing" in the Proceedings of ASP-DAC 2024.

[[Link Missing]]()

## File Structure

```
.
├── Stuff              // DatasetLoader and libraries necesarry for running model
├── images             // Different image results of the models
├── models             // Implementation of ML models - Baselineas + KalmanHD
├── preprocessed_data  // Datasets data
├── scripts            // Collection of bash and .py scripts for various experiments in the paper
├── README.md          // This file
├── main.py            // Main script that runs and tests each model
├── plot.ipynb         // Notebook to plot the accuracy results of the different models in the 3 different types of noise
├── plot_eff.ipynb     // Notebook to plot the running times results of the different models
└── requirements.txt   // Prerequisites
```
## Prerequisites

We test with Python3.9. We recommend using conda environments:

Steps to setup the repository:
   1) Clone this repository.
      ```bash
      git clone https://github.com/DarthIV02/KalmanHD
      ```
   3) Create a new anaconda environment and activate.
      ```bash
      conda create -n hd_forecasting python=3.9
      conda activate hd_forecasting
      ```
   5) Run the following commands to install python packages:
      ```bash
      pip install -r requirements.txt
      pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
      ```
All Python packages required are included in `requirements.txt` and can be installed automatically except for torch with CUDA (in the case of running script in GPU).

### Dataset Preparation

As mentioned in the paper, we experiment on Energy Consumption Fraunhoufer, San Francisco Traffic, Metro Interstate Traffic Volume, Guangzhao Traffic and Electricity Load Diagrams from [Tzagkarakis, C. et. al](https://github.com/pcharala/multiple-timeseries-forecasting/tree/master). 

* All the dataset data files can be found in `./preprocessed_data` folder (most of them are .csv files).
* The datasets are loaded into the main script with the help of `./Stuff/DatasetLoader.py` file.

## Getting Started

To run KalmanHD without any type of noise in the dataset SanFranciscoTraffic:

```bash
python main.py --model KalmanHD --dataset SanFranciscoTraffic
```
To run any other baseline, run the past command with the attribute "--model" changed:

* `RegHD`: Implementation of RegHD in [RegHD: Robust and Efficient Regression in Hyper-Dimensional Learning System](https://ieeexplore.ieee.org/document/9586284), DAC '21
* `VAE`: Implementation of PFVAE in [PFVAE: a planar flow-based variational auto-encoder prediction model for time series data](https://www.mdpi.com/2227-7390/10/4/610), Mathematics 2022
* `DNN`: Implementation of E-Sense in [A diverse noise-resilient dnn ensemble model on edge devices for time-series data](https://ieeexplore.ieee.org/document/9491607), SECON '21
* `KalmanFilter`: Implementation of online Kalman Filter in [Autoregressive-model-based methods for online time series prediction with missing values: an experimental evaluation](https://arxiv.org/abs/1908.06729)

To run any another dataset, run the past command with the attribute "--dataset" changed:

* `SanFranciscoTraffic`: San Francisco Traffic [default if no dataset specified]
* `MetroInterstateTrafficVolume`: Metro Interstate Traffic Volume
* `GuangzhouTraffic`: Guangzhao Traffic
* `EnergyConsumptionFraunhofer`: Energy Consumption Fraunhoufer
* `ElectricityLoadDiagrams`: Electricity Load Diagrams

To change the learning rate in the HD-based methods change the "--learning_rate" attribute with the float-value desired as shown below (default: 0.0001):

```bash
python main.py --model KalmanHD --learning_rate 0.0001
```
We list other hyperparameters that can be modified: 
* `--dimension_hd`: To change the dimension of the hypervectors in the HD-based methods with the int-value desired (default: 1000)
* `--hd_encoder`: To change the encoder function in the HD-based methods (default: 'nonlinear') other options include:
   * 'linear': Linear-based encoding
   * 'time_encoding': TimeSeries-based encoding, with permutation on each time stamp
* `--print_freq`: How often to test the current model with the cross-validation data (default: 1000)
* `--epochs`: Number of training epochs or number of passes on dataset (default: 1). All of the experiments are set to 1, given that we test on the online setting.
* `--levels`: Number of levels to divide encoding when using a linear hd-encoder for HD-based methods (default: 6)
* `--size_of_sample`: Number of previous samples before forecasting (default: 20)
* `--alpha`: Percentage of importance of the past variance in the moving average variance (default: 0.3)
* `--device`: Device to use if possible, either cpu or gpu (default: 'gpu')

We list the arguments that can be modified for the different types of noises:

* Gaussian Noise:
   *  `--gaussian_noise`: Standard deviation of the gaussian noise (default: 0.0). Change this parameter to any other float-value to add gaussian noise.
* Poisson Noise:
   *  `--poisson_noise`: Lambda for the poisson noise (default: 0.0). Change this parameter to any other float-value to add poisson noise.
*  Missing Values:
   * `--s`: Number of consecutive missing values (default: 5)
   * `--p`: Percentage of probability of value missing (default: 0.0). Change this parameter to any other float-value between (0-1) to add missing values. The higher the porcentage, the higher number of missing values in the input training dataset.

### Scripts

We provide our scripts for running various experiments in the paper in `scripts`:

```
.
├── run_exp.py                           // Python script for running baselines in all of the datasets with the hyperparameters set in the paper
├── run_exp.sh                           // Bash script for running baselines in all of the datasets with the hyperparameters set in the paper
├── run_exp_gaussian.py                  // Python script for running all of the methods with Gaussian noise added
├── run_exp_gaussian.sh                  // Bash script for running all of the methods with Gaussian noise added
├── run_exp_missing.py                   // Python script for running all of the methods with different levels of missing data
├── run_exp_missing.sh                   // Bash script for running all of the methods with different levels of missing data
├── run_exp_poisson.sh                   // Bash script for running all of the methods with Possion noise added
└── run_test_lr.sh                       // Bash script for testing different levels of learning rate in KalmanHD on all datasets
```
