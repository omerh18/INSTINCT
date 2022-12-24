from experiments.architucture_study_experiments.hyper_params_study import run_experiment_configurations
from experiments.experiments_utils import RESULTS_DIR
from utils import *
import pandas as pd


DEFAULT_AGG = 'mean'


def get_model_configurations_results_summary(gs_results_df, agg_func=DEFAULT_AGG):
    gs_results_splits_df = gs_results_df.groupby(['dataset', 'kernel_sizes', 'depth', 
                                                  'use_residual', 'bottleneck_size', 
                                                  'num_classifiers', 'number_of_filters', 
                                                  'global_pool', 'split_index']).agg({'test_accuracy': agg_func, 
                                                                                      'test_AUC': agg_func}).reset_index()
    
    gs_params_results_df = gs_results_splits_df.groupby(['dataset', 'kernel_sizes', 'depth', 
                                                         'use_residual', 'bottleneck_size', 
                                                         'num_classifiers', 'number_of_filters',
                                                         'global_pool']).agg({'test_accuracy': agg_func, 
                                                                              'test_AUC': agg_func}).reset_index()
    
    return gs_params_results_df


def run_pooling_benchmark_experiment(gs_model_configurations, experiment_name, 
									 min_num_classifiers=1, max_num_classifiers=32):
    gs_results_df = run_experiment_configurations(gs_model_configurations, min_num_classifiers, max_num_classifiers)
    
    gs_params_results_df = get_model_configurations_results_summary(gs_results_df)
    
    gs_params_results_df = gs_params_results_df.rename({'global_pool': 'pooling'}, axis=1)
    
    pooling_dict = {False: 'FAP', True: 'GAP'}
    gs_params_results_df['pooling'] = gs_params_results_df['pooling'].map(pooling_dict) 
    
    gs_params_results_df.to_csv(f'{RESULTS_DIR}/{experiment_name}.csv')
    
    return gs_params_results_df
