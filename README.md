# KalmanHD: Robust On-Device Time Series Forecasting with Hyperdimensional Computing

This repo contains the simulation implementation for paper:

Ivannia Gomez, Xiaofan Yu, Tajana Rosing. "KalmanHD: Robust On-Device Time Series Forecasting with Hyperdimensional Computing" in the Proceedings of ASP-DAC 2024.

[arXiv link]

Steps to setup the code:
   1) Clone this repository.
      ```bash
      git clone https://github.com/DarthIV02/TimeSeriesForecasting-with-HD
      ```
   3) Create a new anaconda environment and activate.
      ```bash
      conda create -n hd_forecasting python=3.9
      conda activate hd_forecasting
      ```
   5) Run the following commands:
      ```bash
      pip install -r requirements.txt
      pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
      ```

Important files:
   - *main.py* is the main script that runs and tests each model.
   - The scripts for our model and the different baselines can be found in the folder *models*.
   - Inside the *Stuff* folder you can finde the *DatasetLoader.py* script, which loads and initializes the datasets.

To run all of the models in all of the datasets for each model run **run_exp.sh**, for missing values test run **run_exp_missing.sh** and for gaussian noise on the datasets run **run_exp_gaussian.sh**
