from experiments.experiments_utils import RESULTS_DIR
from classifier import SingleINSTINCTClassifier, get_scores
from utils import *
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd


def run_model_dataset_configurations(datasets_dir, dataset_configurations, model_configurations, num_epochs=100, iteration_idx=0):
	experimental_results = []

	for dataset_configuration in dataset_configurations:
		
		print(f'Dataset configuration: {dataset_configuration}')
		
		X_train, y_train, X_test, y_test = create_synthetic_discriminative_dataset(datasets_dir, **dataset_configuration)
		
		y_train_softmax, y_test_softmax = to_categorical(y_train), to_categorical(y_test)
		
		num_classes = len(np.unique(y_train))
		
		for model_configuration in model_configurations:
		
			print(f'Model configuration: {model_configuration}')
		
			model = SingleINSTINCTClassifier(input_shape=X_train.shape[1:],
											 dataset_name='synthetic_hp',
											 num_classes=num_classes,  
											 stride=1,
											 use_residual=True,
											 bottleneck_size=None, 
											 add_checkpoint=True,
											 print_summary=True,
											 **model_configuration)

			hist = model.fit(X_train,
							 y_train_softmax,
							 X_test,
							 y_test_softmax,
							 batch_size=None, 
							 epochs=num_epochs,
							 verbose=True) 
			
			y_test_pred, y_test_pred_probas = model.predict(X_test) 

			test_accuracy, test_auc = get_scores(y_test, y_test_pred, y_test_pred_probas, num_classes)
			
			y_train_pred, y_train_pred_probas = model.predict(X_train) 

			train_accuracy, train_auc = get_scores(y_train, y_train_pred, y_train_pred_probas, num_classes)
			
			iteration_results = {'train_accuracy': train_accuracy, 
								 'train_AUC': train_auc,
								 'test_accuracy': test_accuracy, 
								 'test_AUC': test_auc}
			
			print(f'Iteration results: {iteration_results}, Iteration index: {iteration_idx}')
			
			iteration_info = dict(dataset_configuration, **model_configuration, **iteration_results)
			
			experimental_results.append(iteration_info)
			
			iteration_idx += 1
			
	return experimental_results

		
def run_network_architecture_study_experiment(datasets_dir, 
											  dataset_configurations, model_configurations, 
                                              pivot_depth, pivot_kernel_size, num_epochs=100):
	experimental_results = run_model_dataset_configurations(datasets_dir, dataset_configurations, 
															model_configurations, num_epochs=num_epochs)

	results_df = pd.DataFrame(experimental_results)
    
	for level in [0, 1]:
        
		level_results_by_filter_length = results_df[(results_df['level'] == level) & 
													(results_df['depth'] == pivot_depth)].reset_index(drop=True)
		level_results_by_filter_length.to_csv(f'{RESULTS_DIR}/architecture_change_filter_length_level_{level}.csv')

		level_results_by_depth = results_df[(results_df['level'] == level) & 
											(results_df['kernel_sizes'] == f'[{pivot_kernel_size}]')].reset_index(drop=True)
		level_results_by_depth.to_csv(f'{RESULTS_DIR}/architecture_change_depth_level_{level}.csv')
		
	return results_df
