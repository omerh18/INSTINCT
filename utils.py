from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import os


DATASETS_DIR = 'ClassificationDatasets'
DATA_SEPARATOR = '\t'


def read_labels_file(dataset):
    labels_file = f'{DATASETS_DIR}/{dataset}/labels.txt'
    
    with open(labels_file, 'r') as f:
        
        labels = f.readlines()
        
        labels = [label.strip().split(DATA_SEPARATOR) for label in labels]
        
        labels = pd.DataFrame([{'entity': int(label[0]), 
                                'label': int(label[1])} for label in labels])    
    
    return labels


def read_data_file(dataset):
    data_file = f'{DATASETS_DIR}/{dataset}/data.txt'
    
    with open(data_file, 'r') as f:
        
        data = f.readlines()
        
        data = [sti.strip().split(DATA_SEPARATOR) for sti in data]
        
        data = pd.DataFrame([{'entity': int(sti[0]), 
                              'symbol': int(sti[1]), 
                              'start': int(sti[2]), 
                              'finish': int(sti[3])} for sti in data]) 
    
    return data	
	
	
def read_specific_dataset(dataset):
    labels = read_labels_file(dataset)

    data = read_data_file(dataset)

    return labels, data, dataset
	
	
def read_all_datasets():
    for d in os.scandir(DATASETS_DIR):
        
        try:
        
            if d.is_dir() and not d.name.startswith('.'): 

                dataset = d.name

                yield read_specific_dataset(dataset)
            
        except:
            
            pass

			
def create_synthetic_dataset(stis_series_len, entities_num, symbols_num, stis_num, labels_num, name):
	path_to_dataset = f'{DATASETS_DIR}/{name}/'
	
	if not os.path.isdir(path_to_dataset):
		os.mkdir(path_to_dataset)

	np.random.seed(0)
	
	data_file = f'{path_to_dataset}data.txt'
	
	with open(data_file, 'w') as f:
	
		for i in tqdm(range(entities_num)):

			for j in range(stis_num):

				symbol = np.random.randint(0, symbols_num)

				start = np.random.randint(0, stis_series_len)

				finish = np.random.randint(start, stis_series_len)

				f.write(f'{i}\t{symbol}\t{start}\t{finish}\n')
            
	labels_file = f'{path_to_dataset}labels.txt'

	labels = np.random.randint(0, labels_num, entities_num)

	with open(labels_file, 'w') as f:
    
		for i in range(entities_num):
			f.write(f'{i}\t{labels[i]}\n')

	print('Finished generating dataset')

	
def create_synthetic_discriminative_dataset(synthetic_discriminative_datasets_dir,
											discriminative_pattern_len=[0.25],
											discriminative_pattern_position=[0.1, 0.65], 
											stis_series_len=128, 
											entities_num=256, 
											symbols_num=32, 
											level=0):
	path_to_datasets_dir = f'{DATASETS_DIR}/{synthetic_discriminative_datasets_dir}/'
	if not os.path.isdir(path_to_datasets_dir):
		os.mkdir(path_to_datasets_dir)
	
	np.random.seed(42)
	
	num_classes = len(discriminative_pattern_len) * len(discriminative_pattern_position)
    
    # create the classes definitions
	classes_def = []
	for pattern_len in discriminative_pattern_len:
		for pattern_position in discriminative_pattern_position:
			classes_def.append({'pattern_len': int(pattern_len * stis_series_len),
								'pattern_pos': int(pattern_position * stis_series_len)})
    
	return_values = []
    
	for i in range(2):

		X = np.random.normal(0, 1, size=(entities_num, stis_series_len, symbols_num))
		X = (X >= 0).astype('int')

		y = np.random.randint(low=0, high=num_classes, size=(entities_num,))
		y[:num_classes] = np.arange(start=0, stop=num_classes, dtype=np.int32)

		# create the dataset
		for i in range(entities_num):
			sample_class = y[i]

			current_pattern_pos = classes_def[sample_class]['pattern_pos']
			currrent_pattern_len = classes_def[sample_class]['pattern_len']

			X[i][current_pattern_pos: current_pattern_pos + currrent_pattern_len] += 2

			if level == 1:
            
				current_pattern_pos = classes_def[1 - sample_class]['pattern_pos']
				currrent_pattern_len = classes_def[1 - sample_class]['pattern_len']

				joker = np.random.randint(0, currrent_pattern_len)

				X[i][current_pattern_pos: current_pattern_pos + joker, :] += 2
				X[i][current_pattern_pos + joker + 1: current_pattern_pos + currrent_pattern_len, :] += 2
            
		return_values.extend([X, y])
        
	with open(f'{DATASETS_DIR}/{synthetic_discriminative_datasets_dir}/e_{entities_num}_t_{stis_series_len}_s_{symbols_num}_level_{level}.pickle', 'wb') as f:
		pickle.dump(return_values, f)

	return return_values
