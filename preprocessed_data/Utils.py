# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:19:45 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

from . import error_metrics

import numpy as np
import tabulate as tb


class Utils(object):

    @staticmethod
    def print_dataset_info(df, name):
        exist_missing_values =  df.isnull().values.any()
        
        print('---------------------------------------------------')
        print('Dataset name: ', name)
        print('Dataset contains missing values? ', exist_missing_values)
    
    
    @staticmethod   
    def print_ts_info(ts_array):
    
        print('---------------------------------------------------')
        print("Dataset shape: {}".format(ts_array.shape))
        print("This dataset has {} series, and each series has {} time steps"
              .format(ts_array.shape[0], ts_array.shape[1])
              )
        print('Max value: {} , Min value: {}'.format(np.amax(ts_array),
                                                     np.amin(ts_array)))


    @staticmethod   
    def get_statistics(results, stats, mcruns=None, metrs=None):
    
        print('\n-------------BEGIN SUMMARY-------------\n')
        print('Dataset name: ', results['dataset'])
        print('Model: ', results['model'])
        print('MC runs:', results['mcruns'])
        for exp_res in results['exp_results']:
            #print('Experimental window: {}\n'.format(exp_res['exp_window']))
            summary_list = list()
            
            if metrs:
                for metric in metrs:
                    v = exp_res['metrics'][metric]
                    if mcruns:
                        v = v[:mcruns]
                    summary_list.append(
                        [exp_res['exp_window'], metric] +
                        [getattr(np, stat)(v) for stat in stats])
                    
            else:
                for metric, v in exp_res['metrics'].items():
                    if mcruns:
                        v = v[:mcruns]
                    summary_list.append(
                        [exp_res['exp_window'], metric] +
                        [getattr(np, stat)(v) for stat in stats])
            
            tab = tb.tabulate(summary_list,
                              headers = ['Exp. window',
                                         'Error metric'] + stats,
                              tablefmt="psql")
            print(tab)
        print('\n--------------END SUMMARY--------------')
        
        
    @staticmethod
    def print_summary_statistics(total_results,
                                 stats):    
        
        if len(total_results)==0:
            return
        
        print('\n-------------BEGIN SUMMARY-------------\n')

        for res in total_results:
            Utils.print_experiment_statistics(res['dataset'],
                                              res['model'],
                                              res['mcruns'],
                                              res['exp_results'],
                                              stats)

        print('\n--------------END SUMMARY--------------')

    @staticmethod   
    def print_experiment_statistics(dataset_name,
                                    model,
                                    mcruns,
                                    exp_results,
                                    stats):
        print('----------------------------------------')
        print('Dataset name: ', dataset_name)
        print('Model: ', model)
        print('MC runs:', mcruns)
        for exp_res in exp_results:
            #print('Experimental window: {}\n'.format(exp_res['exp_window']))
            summary_list = list()
            for metric, v in exp_res['metrics'].items():
                summary_list.append([exp_res['exp_window'], metric] +
                                    [getattr(np, stat)(v) for stat in stats])
            
            tab = tb.tabulate(summary_list,
                              headers = ['Exp. window',
                                         'Error metric'] + stats,
                              tablefmt="psql")
            print(tab)

        print('----------------------------------------')


    @classmethod
    def compute_pred_error(self, original, prediction, error_type):
       try:
           return error_metrics.METRICS[error_type](original, prediction)
       except:
           raise ValueError('Unknown error type: {0}'.format(error_type))
