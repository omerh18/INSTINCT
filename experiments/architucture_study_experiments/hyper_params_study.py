from experiments.experiments_utils import RESULTS_DIR
from classifier import EnsembleINSTINCTClassifier, get_scores
import representation_generator
from utils import *
from sklearn.model_selection import train_test_split, KFold
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd


datasets_configurations = {
    'blocks': {
        'epochs': 100, 
        'strides': [1]
    },
    'pioneer': {
        'epochs': 100, 
        'strides': [1]
    },
    'context': {
        'epochs': 100, 
        'strides': [1]
    },
    'auslan2': {
        'epochs': 100, 
        'strides': [1]
    },
    'skating': {
        'epochs': 100, 
        'strides': [10]
    },
    'hepatitis': {
        'epochs': 100, 
        'strides': [10]
    },
}


DEFAULT_AGG = 'mean'


def get_classifier_fold_predictions(model, X_train, y_train, X_test, y_test, num_classes, selected_classifiers):
    y_test_pred, y_test_pred_probas = model.predict(X_test, selected_classifiers)

    test_accuracy, test_auc = get_scores(y_test, y_test_pred, y_test_pred_probas, num_classes)

    y_train_pred, y_train_pred_probas = model.predict(X_train, selected_classifiers) 

    train_accuracy, train_auc = get_scores(y_train, y_train_pred, y_train_pred_probas, num_classes)

    iteration_results = {'train_accuracy': train_accuracy, 
                         'train_AUC': train_auc,
                         'test_accuracy': test_accuracy, 
                         'test_AUC': test_auc}

    return iteration_results
	

def get_models_prediction_results(max_num_classifiers, classifiers_options, gs_model_configurations, cv_n_splits=10):
    grid_search_experimental_results = []
    
    iteration_idx = 0
    
    kf = KFold(n_splits=cv_n_splits)
    
    for labels_df, data_df, dataset in read_all_datasets():
        
        if dataset not in datasets_configurations.keys():
            continue
    
        print(f'Now processing {dataset}')

        rep1d, entities_labels, _ = representation_generator.generate_representation(labels_df, data_df)

        num_classes = labels_df['label'].nunique()

        num_epochs, strides = datasets_configurations[dataset]['epochs'], datasets_configurations[dataset]['strides']

        for stride in strides:

            print(f'Stride {stride}')

            dataset_configuration = {'dataset': dataset, 'stride': stride}

            for model_configuration in gs_model_configurations:

                print(f'Model configuration: {model_configuration}')

                split_idx = 0

                for train_index, test_index in kf.split(rep1d):

                    X_train, X_test = rep1d[train_index], rep1d[test_index]

                    y_train, y_test = entities_labels[train_index], entities_labels[test_index]

                    if num_classes > min(np.unique(y_train).shape[0], np.unique(y_test).shape[0]):
                        split_idx += 1
                        iteration_idx += len(classifiers_options)
                        continue

                    y_train_softmax, y_test_softmax = to_categorical(y_train), to_categorical(y_test)

                    model = EnsembleINSTINCTClassifier(num_classifiers=max_num_classifiers, 
													   input_shape=X_train.shape[1:], 
													   dataset_name=dataset,
													   num_classes=num_classes, 
													   stride=stride,
													   add_checkpoint=True,
													   print_summary=True, 
													   **model_configuration)

                    model.fit(X_train,
                              y_train_softmax,
                              X_test,
                              y_test_softmax,
                              batch_size=None, 
                              epochs=num_epochs,
                              verbose=True)

                    for selected_classifiers in classifiers_options:
                        num_classifiers = selected_classifiers.count('1')

                        iteration_results = get_classifier_fold_predictions(model, X_train, y_train, X_test, y_test, 
                                                                            num_classes, selected_classifiers)
                        
                        print(f'Iteration results: {iteration_results}, Iteration index: {iteration_idx}')

                        iteration_info = dict(dataset_configuration, **model_configuration, 
                                              **{'num_classifiers': num_classifiers, 
                                                 'selected_classifiers': selected_classifiers,
                                                 'split_index': split_idx}, 
                                              **iteration_results)

                        grid_search_experimental_results.append(iteration_info)

                        iteration_idx += 1

                    split_idx += 1
                    
    return grid_search_experimental_results
	

def get_model_configurations_results_summary(gs_results_df, agg_func=DEFAULT_AGG):
    gs_results_splits_df = gs_results_df.groupby(['dataset', 'kernel_sizes', 'depth', 
                                                  'use_residual', 'bottleneck_size', 
                                                  'num_classifiers', 'number_of_filters', 
                                                  'split_index']).agg({'test_accuracy': agg_func, 
                                                                       'test_AUC': agg_func}).reset_index()
    
    gs_params_results_df = gs_results_splits_df.groupby(['dataset', 'kernel_sizes', 'depth', 
                                                         'use_residual', 'bottleneck_size', 
                                                         'num_classifiers', 
                                                         'number_of_filters']).agg({'test_accuracy': agg_func, 
                                                                                    'test_AUC': agg_func}).reset_index()
    
    return gs_params_results_df


def run_experiment_configurations(gs_model_configurations, min_num_classifiers=1, max_num_classifiers=32):
    classifiers_options = [((num_classifiers + 1) * '1').zfill(max_num_classifiers) 
                           for num_classifiers in range(min_num_classifiers - 1, max_num_classifiers)]
    
    grid_search_experimental_results = get_models_prediction_results(max_num_classifiers, 
                                                                     classifiers_options, 
                                                                     gs_model_configurations)
    
    gs_results_df = pd.DataFrame(grid_search_experimental_results)
    
    gs_results_df['kernel_sizes'] = gs_results_df['kernel_sizes'].astype('str')

    gs_results_df['bottleneck_size'] = gs_results_df['bottleneck_size'].fillna(-1)
    
    return gs_results_df
	
	
# (K) BEST HYPER-PARAMS GRID SEARCH 


def get_best_overall_scores(gs_params_results_df):
    accuracy_scores = gs_params_results_df.groupby(['dataset'])['test_accuracy'].max()

    auc_scores = gs_params_results_df.groupby(['dataset'])['test_AUC'].max()

    best_scores = pd.concat([accuracy_scores, auc_scores], axis=1) 
    
    return best_scores


def get_best_k_configs_by_objective(config_overall_mean_scores, gs_params_results_df, objective, k):
    config_key = ['kernel_sizes', 'depth', 'use_residual', 'bottleneck_size', 'num_classifiers', 'number_of_filters']
    
    configs_sorted_by_objective = config_overall_mean_scores.sort_values(objective, ascending=False).reset_index().reset_index()
    
    best_k_configs = configs_sorted_by_objective[configs_sorted_by_objective['index'] < k][config_key]
    
    best_k_configs_scores = best_k_configs.merge(gs_params_results_df[config_key + ['dataset', objective]], 
                                                 left_on=config_key, right_on=config_key)
    
    return best_k_configs_scores


def get_best_scores(gs_params_results_df, k=5):
    best_overall_scores = get_best_overall_scores(gs_params_results_df)
    
    config_overall_mean_scores = gs_params_results_df.groupby(['kernel_sizes', 'depth', 
                                                               'use_residual', 'bottleneck_size', 
                                                               'num_classifiers', 
                                                               'number_of_filters']).agg({'test_accuracy': 'mean', 
                                                                                          'test_AUC': 'mean'})
    
    best_accuracy_configs = get_best_k_configs_by_objective(config_overall_mean_scores, gs_params_results_df, 'test_accuracy', k)
    
    best_auc_configs = get_best_k_configs_by_objective(config_overall_mean_scores, gs_params_results_df, 'test_AUC', k)
    
    return best_scores, best_accuracy_configs, best_auc_configs
	
	
def run_hp_tuning_experiment(gs_model_configurations, experiment_name, max_num_classifiers):
    gs_results_df = run_experiment_configurations(gs_model_configurations, max_num_classifiers=max_num_classifiers)
    
    gs_params_results_df = get_model_configurations_results_summary(gs_results_df)
    
    best_overall_scores, best_accuracy_configs, best_auc_configs = get_best_scores(gs_params_results_df, k=5)
    
    best_overall_scores.to_csv(f'{RESULTS_DIR}/{experiment_name}_best_overall_scores.csv')
    
    best_accuracy_configs.to_csv(f'{RESULTS_DIR}/{experiment_name}_best_accuracy_configs.csv')
    
    best_auc_configs.to_csv(f'{RESULTS_DIR}/{experiment_name}_best_auc_configs.csv')
    
    return best_overall_scores, best_accuracy_configs, best_auc_configs
	

# SINGLE HYPER-PARAM ANALYSIS 
	
	
def run_single_hp_analysis_experiment(gs_model_configurations, experiment_name, 
									  min_num_classifiers=1, max_num_classifiers=32):
    gs_results_df = run_experiment_configurations(gs_model_configurations, min_num_classifiers, max_num_classifiers)
    
    gs_params_results_df = get_model_configurations_results_summary(gs_results_df)
    
    gs_params_results_df.to_csv(f'{RESULTS_DIR}/{experiment_name}.csv')
    
    return gs_params_results_df
