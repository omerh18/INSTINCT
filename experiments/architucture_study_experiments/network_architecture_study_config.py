STIS_SERIES_LENGTHS = [128, 256, 512, 1024, 2048]
DEFAULT_PATTERN_LEN = 0.25
DEFAULT_PATTERN_POSITIONS = [0.1, 0.65]
DEFAULT_ENTITIES_NUM = 1024
DEFAULT_SYMBOLS_NUM = 8
DEFAULT_PIVOT_DEPTH = 3
DEFAULT_PIVOT_KERNEL_SIZE = 20


dataset_configurations = []


for level in [0, 1]:
    
    dataset_configurations.extend([{'discriminative_pattern_len': [DEFAULT_PATTERN_LEN], 
                                    'discriminative_pattern_position': DEFAULT_PATTERN_POSITIONS, 
                                    'stis_series_len': stis_series_len, 
                                    'entities_num': DEFAULT_ENTITIES_NUM, 
                                    'symbols_num': DEFAULT_SYMBOLS_NUM, 
                                    'level': level} for stis_series_len in STIS_SERIES_LENGTHS])

									
model_configurations = []


for kernel_size in [5, 10, 20, 30, 40, 60, 80, 100]:
        
	for global_pool in [False, True]: 
		
		model_configurations.append({'kernel_sizes': [kernel_size], 
									 'global_pool': global_pool, 
									 'depth': DEFAULT_PIVOT_DEPTH})
										 
 
for depth in [1, 2, 3, 4, 5, 7]:
	
	for global_pool in [False, True]: 
		
		model_configurations.append({'kernel_sizes': [DEFAULT_PIVOT_KERNEL_SIZE], 
									 'global_pool': global_pool, 
									 'depth': depth})
