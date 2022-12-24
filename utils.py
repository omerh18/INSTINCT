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
