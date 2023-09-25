# KalmanHD: Robust On-Device Time Series Forecasting with Hyperdimensional Computing

This repo contains the simulation implementation for paper:

Ivannia Gomez, Xiaofan Yu, Tajana Rosing. "KalmanHD: Robust On-Device Time Series Forecasting with Hyperdimensional Computing" in the Proceedings of ASP-DAC 2024.

[[arXiv link]]()

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
   5) Run the following commands to install packages:
      ```bash
      pip install -r requirements.txt
      pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
      ```
All Python packages required are included in `requirements.txt` and can be installed automatically except for torch with CUDA, in the case of running script in GPU.

### Dataset Preparation

As mentioned in the paper, we experiment on Energy Consumption Fraunhoufer, San Francisco Traffic, Metro Interstate Traffic Volume, Guangzhao Traffic and Electricity Load Diagrams from [Tzagkarakis, C. et. al](https://github.com/pcharala/multiple-timeseries-forecasting/tree/master). 

* All the dataset data files can be found in `./preprocessed_data` folder (most of them are .csv files).
* The datasets are loaded into the main script with the help of `./Stuff/DatasetLoader.py` file.

## Getting Started

### Scripts

We provide our scripts for running various experiments in the paper in `scripts`:

```
.
├── run_exp.py                           // Python script for running baselines in all of the datasets with the hyperparameters set in the paper
├── run_exp.sh                           // Bash script for running baselines in all of the datasets with the hyperparameters set in the paper
├── run_baseline.sh                      // Script for running baselines in the random delay setup
├── run_exp_nycmesh.sh                   // Script for running Async-HFL in the NYCMesh setup
├── run_exp.sh                           // Script for running Async-HFL in the random delay setup
├── run_motivation_nycmesh.sh            // Script for running the motivation study
├── run_sensitivity_pca.sh               // Script for running sensitivity study regarding PCA dimension
└── run_sensitivity_phi.sh               // Script for running sensitivity study regarding phi
```
To run all of the models in all of the datasets for each model run **run_exp.sh**, for missing values test run **run_exp_missing.sh** and for gaussian noise on the datasets run **run_exp_gaussian.sh**
