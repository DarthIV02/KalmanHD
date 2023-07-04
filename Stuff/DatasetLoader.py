# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:35 2022

@author: Ivannia Gomez <ivannia.gomez@cetys.edu.mx>
"""

import pandas as pd
import numpy as np

from preprocessed_data.data_loader_Monash import convert_tsf_to_dataframe
from preprocessed_data.Utils import Utils


class DatasetLoader(object):
    """
    A class trat represents a dataset loader.

    Attributes
    ----------
    dataset_name: str
        the name of the dataset.
    dataset_path: str
        the dataset path.
    dataset_type: stt
        Modify if its normalized or not
    """

    def __init__(self,
                 dataset_name):

        self._dataset_name = dataset_name

    def dataset_load_and_preprocess(self, dataset_type="original"):
        """
        Loads, preprocesses and returns dataset as ndarray.

        Raises
        ------
        ValueError
            If unknow dataset name.

        Returns
        -------
        ndarray
            Rows correspond to timeseries and columns to time steps.

        """

        if self._dataset_name == 'SanFranciscoTraffic':

            self._dataset_path = 'preprocessed_data/SanFranciscoTraffic/traffic_weekly_dataset.tsf'

            (df,
             frequency,
             forecast_horizon,
             contain_missing_values,
             contain_equal_length) = \
                convert_tsf_to_dataframe(self._dataset_path)

            original_ts = \
                np.zeros((df.shape[0], len(df.iloc[0, 2].to_numpy())))
            for col in range(df.shape[0]):
                tmp_series = (df.iloc[col, 2]).to_numpy().reshape(-1, 1)
                original_ts[col, :] = np.transpose(tmp_series)

        elif self._dataset_name == 'GuangzhouTraffic':

            self._dataset_path = 'preprocessed_data/GuangzhouTraffic/traffic_speed_hourly.csv'

            df = pd.read_csv(self._dataset_path, header=None)
            original_ts = df.to_numpy()

        elif self._dataset_name == 'ElectricityLoadDiagrams':

            self._dataset_path = 'preprocessed_data/ElectricityLoadDiagrams/electr_daily.csv'

            df = pd.read_csv(self._dataset_path)

            # Drop the columns that do not contain any valuable information
            df.drop(df.columns[[0, 1, 322, 323, 324]],
                    axis=1,
                    inplace=True)

            # Convert the dataframe into a 2D numerical array
            orig_ts = df.to_numpy()

            # Rows should correspond to the time-series and columns correspond to the time-stamps
            original_ts = np.transpose(orig_ts)

        elif self._dataset_name == 'EnergyConsumptionFraunhofer':

            self._dataset_path = 'preprocessed_data/EnergyConsumptionFraunhofer/electrFraunhofer_daily.csv'

            df = pd.read_csv(self._dataset_path)

            # Remove first dataframe's column since it contains the dates
            df = df.iloc[:, 1:]

            # Convert the dataframe into a 2D numerical array
            orig_ts = df.to_numpy()

            # Rows should correspond to the time-series and columns correspond to the time-stamps
            original_ts = np.transpose(orig_ts)

        elif self._dataset_name == 'LondonSmartMeters':

            self._dataset_path = 'preprocessed_data/LondonSmartMeters/London_elec_HalfHourly.csv'

            df = pd.read_csv(self._dataset_path)

            # Select dates in the range below
            # 15024    2012-10-23 00:00:01
            # 25007    2013-05-18 23:30:01
            df = df.iloc[15024:25007, :]
            del df["Unnamed: 0"]
            del df["timestamps"]

            # Convert the dataframe into a 2D numerical array
            orig_ts = df.to_numpy()

            # Rows should correspond to the time-series and columns correspond to the time-stamps
            original_ts = np.transpose(orig_ts)

        elif self._dataset_name == "MetroInterstateTrafficVolume":
            
            self._dataset_path = 'preprocessed_data/MetroInterstateTrafficVolume/Metro_Interstate_Traffic_Volume.csv.gz'

            df = pd.read_csv(self._dataset_path)
            original_ts = np.array(df.traffic_volume)
            original_ts = original_ts.reshape(1, original_ts.shape[0])

        else:

            raise ValueError('Unknown dataset: ' + self._dataset_name)

        Utils.print_dataset_info(df, self._dataset_name)
        # Utils.print_ts_info(original_ts)
        print("--------------------------")

        if (dataset_type == "normalized"):
            matrix_sc = np.empty(
                [original_ts.shape[0], original_ts.shape[1]])
            for i in range(original_ts.shape[0]):  # for each time series
                # Check that they're diffrent values
                if (original_ts[i].max() != original_ts[i].min()):
                    y_sc = original_ts[i] - original_ts[i].min()
                    y_sc = y_sc / \
                        (original_ts[i].max() - original_ts[i].min())
                else:
                    y_sc = original_ts[i]
                matrix_sc[i] = y_sc
            original_ts = matrix_sc
        return original_ts
