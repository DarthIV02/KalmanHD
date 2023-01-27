# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:35 2022

@author: Ivannia Gomez <ivannia.gomez@cetys.edu.mx>
"""

import pandas as pd
import numpy as np

from multipletimeseriesforecasting.common.data_loader_Monash import convert_tsf_to_dataframe
from multipletimeseriesforecasting.common.Utils import Utils


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
                 dataset_name,
                 dataset_path=""):

        self._dataset_name = dataset_name
        self._dataset_path = dataset_path

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

            df = pd.read_csv(self._dataset_path, header=None)
            original_ts = df.to_numpy()

        elif self._dataset_name == 'ElectricityLoadDiagrams':

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

            df = pd.read_csv(self._dataset_path)

            # Remove first dataframe's column since it contains the dates
            df = df.iloc[:, 1:]

            # Convert the dataframe into a 2D numerical array
            orig_ts = df.to_numpy()

            # Rows should correspond to the time-series and columns correspond to the time-stamps
            original_ts = np.transpose(orig_ts)

        elif self._dataset_name == 'LondonSmartMeters':

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

        elif self._dataset_name == "MetroInterstateTrafficVolumeComplete":

            df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv.gz')
            df['date_time'] = pd.to_datetime(df['date_time'])
            df.drop_duplicates('date_time', inplace=True)
            df.set_index('date_time', inplace=True)
            df = df.reindex(pd.date_range(
                df.head(1).index[0], df.tail(1).index[0], freq='H'))

            df = df[df.index.year.isin([2016, 2017, 2018])].copy()

            df = pd.concat([df.select_dtypes(include=['object']).fillna(method='backfill'),
                            df.select_dtypes(include=['float']).interpolate()], axis=1)

            map_col = dict()

            X = df.select_dtypes(include=['object']).copy()
            for i, cat in enumerate(X):
                X[cat] = df[cat].factorize()[0]
                map_col[cat] = i

            X['month'] = df.index.month
            i += 1
            map_col['month'] = i
            X['weekday'] = df.index.weekday
            i += 1
            map_col['weekday'] = i
            X['hour'] = df.index.hour
            i += 1
            map_col['hour'] = i

            def gen_seq(id_df, seq_length, seq_cols):

                data_matrix = id_df[seq_cols]
                num_elements = data_matrix.shape[0]

                for start, stop in zip(range(0, num_elements-seq_length, 1), range(seq_length, num_elements, 1)):

                    yield data_matrix[stop-sequence_length:stop].values.reshape((-1, len(seq_cols)))

            sequence_length = 24*7

            sequence_input = []
            sequence_target = []

            for seq in gen_seq(X, sequence_length, X.columns):
                sequence_input.append(seq)

            for seq in gen_seq(df, sequence_length, ['traffic_volume']):
                sequence_target.append(seq)

            sequence_input = np.asarray(sequence_input)
            sequence_target = np.asarray(sequence_target)

            if (dataset_type == "normalized"):
                dif = sequence_target.max()-sequence_target.min()
                sequence_target = (sequence_target-sequence_target.min())/dif

            original_ts = np.concatenate(
                (sequence_input, sequence_target), axis=2)

        elif self._dataset_name == "MetroInterstateTrafficVolumeRegression":

            df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv.gz')
            df['date_time'] = pd.to_datetime(df['date_time'])
            df.drop_duplicates('date_time', inplace=True)
            df.set_index('date_time', inplace=True)
            df = df.reindex(pd.date_range(
                df.head(1).index[0], df.tail(1).index[0], freq='H'))

            df = df[df.index.year.isin([2016, 2017, 2018])].copy()

            df = pd.concat([df.select_dtypes(include=['object']).fillna(method='backfill'),
                            df.select_dtypes(include=['float']).interpolate()], axis=1)

            a = np.append([0], df.traffic_volume.to_numpy()[:-1])

            d = {'date_time': df.index.to_numpy(), 'traffic_volume_past': a}
            X = pd.DataFrame(data=d)
            X.drop_duplicates('date_time', inplace=True)
            X.set_index('date_time', inplace=True)

            map_col = dict()

            for i, cat in enumerate(X):
                map_col[cat] = i

            def gen_seq(id_df, seq_length, seq_cols):

                data_matrix = id_df[seq_cols]
                print(data_matrix.iloc[0])
                num_elements = data_matrix.shape[0]
                print(num_elements)

                for start, stop in zip(range(0, num_elements-seq_length, 1), range(seq_length, num_elements, 1)):

                    yield data_matrix[stop-sequence_length:stop].values.reshape((-1, len(seq_cols)))

            sequence_length = 24*7

            sequence_input = []
            sequence_target = []

            for seq in gen_seq(X, sequence_length, X.columns):
                sequence_input.append(seq)

            for seq in gen_seq(df, sequence_length, ['traffic_volume']):
                sequence_target.append(seq)

            sequence_input = np.asarray(sequence_input)
            sequence_target = np.asarray(sequence_target)

            if (dataset_type == "normalized"):
                dif = sequence_target.max()-sequence_target.min()
                sequence_target = (sequence_target-sequence_target.min())/dif

            original_ts = np.concatenate(
                (sequence_input, sequence_target), axis=2)

        else:

            raise ValueError('Unknown dataset: ' + self._dataset_name)

        Utils.print_dataset_info(df, self._dataset_name)
        # Utils.print_ts_info(original_ts)
        print("--------------------------")

        if (self._dataset_name != "MetroInterstateTrafficVolumeRegression" and self._dataset_name != "MetroInterstateTrafficVolumeComplete"):
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

        else:
            return sequence_input, sequence_target, map_col, sequence_length, X
