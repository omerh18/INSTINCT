from experiments.experiments_utils import RESULTS_DIR
from classifier import EnsembleINSTINCTClassifier
import representation_generator
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import time


def run_representation_generation_per_sample(labels_df, data_df, sampling_parameter, 
											 data_sampling_dict, experiment_name, iterations=None):
    sampling_values = data_sampling_dict[sampling_parameter]['values']
    
    sampling_parameter_description = data_sampling_dict[sampling_parameter]['description']
    
    iterations = len(sampling_values) if iterations is None else iterations
    
    pp_func = data_sampling_dict[sampling_parameter]['preprocess_function']

    if pp_func is not None:
        pp_args = data_sampling_dict[sampling_parameter]['preprocess_args']
        labels_df, data_df = pp_func(labels_df, data_df, **pp_args)
    
    results = []
    
    for n_samples in sampling_values[: iterations]:
        
        sampled_labels_df, sampled_data_df = data_sampling_dict[sampling_parameter]['sampling_function'](labels_df, 
                                                                                                         data_df, 
                                                                                                         n_samples)

        rep1d, entities_labels, generation_duration = \
			representation_generator.generate_representation(sampled_labels_df, sampled_data_df)

        generation_duration = round(generation_duration, 3)

        print(f'{sampling_parameter_description}: {n_samples}, RGT: {generation_duration}')
        
        results.append({'n_samples': n_samples, 'RGT': generation_duration})
    
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(f'{RESULTS_DIR}/{experiment_name}.csv')
    
    return results_df


def run_training_per_sample(labels_df, data_df, sampling_parameter, data_sampling_dict, experiment_name,
							model_configuration, strides=[1, 10], num_epochs=100, iterations=None, batch_size=None):
    sampling_values = data_sampling_dict[sampling_parameter]['values']
    
    sampling_parameter_description = data_sampling_dict[sampling_parameter]['description']
    
    iterations = len(sampling_values) if iterations is None else iterations 
    
    pp_func = data_sampling_dict[sampling_parameter]['preprocess_function']

    if pp_func is not None:
        pp_args = data_sampling_dict[sampling_parameter]['preprocess_args']
        labels_df, data_df = pp_func(labels_df, data_df, **pp_args)
    
    results = []
    
    for n_samples in sampling_values[: iterations]:

        sampled_labels_df, sampled_data_df = data_sampling_dict[sampling_parameter]['sampling_function'](labels_df, 
                                                                                                         data_df, 
                                                                                                         n_samples)

        rep1d, entities_labels, generation_duration = \
			representation_generator.generate_representation(sampled_labels_df, sampled_data_df)

        X_train, y_train = rep1d, entities_labels

        y_train_softmax = to_categorical(y_train)

        num_classes = len(np.unique(y_train))
		
        for stride in strides:
		
            model = EnsembleINSTINCTClassifier(num_classifiers=1,
										       input_shape=X_train.shape[1:], 
										       dataset_name=None, 
										       num_classes=num_classes, 
										       stride=stride,
										       add_checkpoint=False, 
										       **model_configuration)

            start_time = time.time()

            model.fit(X_train, 
                      y_train_softmax,
                      batch_size=batch_size, 
                      epochs=num_epochs,
                      verbose=False)

            end_time = time.time()

            duration = round(end_time - start_time, 3)
        
            epoch_duration = round(duration / num_epochs, 3)

            print(f'{sampling_parameter_description}: {n_samples}, Training Time: {duration}, Epoch Time: {epoch_duration}, Stride: {stride}')
        
            results.append({'n_samples': n_samples, 'training_time': duration, 'epoch_time': epoch_duration, 'stride': stride})
        
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(f'{RESULTS_DIR}/{experiment_name}.csv')
    
    return results_df
