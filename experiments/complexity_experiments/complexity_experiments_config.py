import numpy as np
import pandas as pd


# RGT = Representation Generation Time
# RGT by Number of entities configuration

RGT_N_SAMPLES_ARRAY = [10, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 2500, 5000, 7500, 10000]

def get_n_entities_of_labels_and_data(labels_df, data_df, n_samples):
    sampled_labels_df = labels_df.sample(frac=1).iloc[: n_samples]
    
    sampled_data_df = data_df.merge(sampled_labels_df, on='entity')
    
    return sampled_labels_df, sampled_data_df

ENTITIES_NUM = 'Number of entities'

ENTITIES_NUM_CODE = 0

# RGT by Number of STIs configuration

RGT_N_STIS_ARRAY = [10, 50, 100, 150, 200, 250]

def get_n_stis_of_labels_and_data(labels_df, data_df, n_stis):
    
    sampled_data_df = data_df.groupby('entity').head(n_stis)
    
    return labels_df, sampled_data_df

STIS_NUM = 'Number of STIs'

STIS_NUM_CODE = 1

# RGT by STIs series length configuration

RGT_STIS_SERIES_LEN_ARRAY = [2**i for i in range(5, 11)]

def get_stis_series_until_specified_len_of_labels_and_data(labels_df, data_df, len_stis_series):
    sampled_data_df = data_df.copy()
    
    sampled_data_df['finish'] = np.minimum(sampled_data_df['finish'], len_stis_series - 1)
    
    trimmed_stis = sampled_data_df[sampled_data_df['start'] >= len_stis_series].shape[0]
    
    random_starts_for_trimmed_stis = np.random.randint(0, len_stis_series, trimmed_stis)
    
    sampled_data_df.loc[sampled_data_df['start'] >= len_stis_series, 'start'] = random_starts_for_trimmed_stis
    
    return labels_df, sampled_data_df

STIS_SERIES_LEN = 'STIs series length'

STIS_SERIES_LEN_CODE = 2

# RGT by Number of symbols configuration

RGT_N_SYMBOLS_ARRAY = [2**i for i in range(3, 8)]

def get_n_symbols_of_labels_and_data(labels_df, data_df, n_symbols):
    sampled_data_df = data_df.copy()
    
    trimmed_stis = sampled_data_df[sampled_data_df['symbol'] >= n_symbols].shape[0]
    
    random_symbols_for_trimmed_stis = np.random.randint(0, n_symbols, trimmed_stis)
    
    sampled_data_df.loc[sampled_data_df['symbol'] >= n_symbols, 'symbol'] = random_symbols_for_trimmed_stis
    
    return labels_df, sampled_data_df

SYMBOLS_NUM = 'Number of symbols'

SYMBOLS_NUM_CODE = 3

# RGT complete configuration

rgt_data_sampling_dict = {
    ENTITIES_NUM_CODE: {
        'description': ENTITIES_NUM,
        'values': RGT_N_SAMPLES_ARRAY, 
        'sampling_function': get_n_entities_of_labels_and_data, 
        'preprocess_function': None,
        'preprocess_args': None 
    }, 
    STIS_NUM_CODE: {
        'description': STIS_NUM,
        'values': RGT_N_STIS_ARRAY, 
        'sampling_function': get_n_stis_of_labels_and_data,
        'preprocess_function': get_n_entities_of_labels_and_data,
        'preprocess_args': {'n_samples': 1024} 
    }, 
    STIS_SERIES_LEN_CODE: {
        'description': STIS_SERIES_LEN,
        'values': RGT_STIS_SERIES_LEN_ARRAY, 
        'sampling_function': get_stis_series_until_specified_len_of_labels_and_data,
        'preprocess_function': get_n_entities_of_labels_and_data,
        'preprocess_args': {'n_samples': 1024}
    },
    SYMBOLS_NUM_CODE: {
        'description': SYMBOLS_NUM,
        'values': RGT_N_SYMBOLS_ARRAY, 
        'sampling_function': get_n_symbols_of_labels_and_data,
        'preprocess_function': get_n_entities_of_labels_and_data,
        'preprocess_args': {'n_samples': 1024}
    }
}

# TT = Training Time
# TT by Number of entities configuration

TT_N_SAMPLES_ARRAY = [10, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 2500, 5000, 7500, 10000]

# TT by Number of STIs configuration

TT_N_STIS_ARRAY = [10, 50, 100, 150, 200, 250]

# TT by STIs series length configuration

TT_STIS_SERIES_LEN_ARRAY = [2**i for i in range(5, 11)]

# TT by Number of symbols configuration

TT_N_SYMBOLS_ARRAY = [2**i for i in range(3, 8)]

# TT complete configuration

tt_data_sampling_dict = {
    ENTITIES_NUM_CODE: {
        'description': ENTITIES_NUM,
        'values': TT_N_SAMPLES_ARRAY, 
        'sampling_function': get_n_entities_of_labels_and_data, 
        'preprocess_function': None,
        'preprocess_args': None 
    }, 
    STIS_NUM_CODE: {
        'description': STIS_NUM,
        'values': TT_N_STIS_ARRAY, 
        'sampling_function': get_n_stis_of_labels_and_data,
        'preprocess_function': get_n_entities_of_labels_and_data,
        'preprocess_args': {'n_samples': 1024} 
    }, 
    STIS_SERIES_LEN_CODE: {
        'description': STIS_SERIES_LEN,
        'values': TT_STIS_SERIES_LEN_ARRAY, 
        'sampling_function': get_stis_series_until_specified_len_of_labels_and_data,
        'preprocess_function': get_n_entities_of_labels_and_data,
        'preprocess_args': {'n_samples': 1024}
    },
    SYMBOLS_NUM_CODE: {
        'description': SYMBOLS_NUM,
        'values': TT_N_SYMBOLS_ARRAY, 
        'sampling_function': get_n_symbols_of_labels_and_data,
        'preprocess_function': get_n_entities_of_labels_and_data,
        'preprocess_args': {'n_samples': 1024}
    }
}
