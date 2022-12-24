# HYPER-PARAMS TUNING

hp_tuning_experiment_name = 'hp_tuning_results' 

hp_tuning_max_num_classifiers = 5

hp_tuning_model_configurations = []

for kernel_sizes in [[5, 10, 15], [10, 20, 30], [20, 30, 40]]:

    for depth in [1, 2, 3, 4, 5]:

        for use_residual in [False, True]: 

            for bottleneck_size in [None, 2, 4, 8, 16, 32]:
                
                for number_of_filters in [4, 8, 16, 32]:

                    hp_tuning_model_configurations.append({'kernel_sizes': kernel_sizes, 
                                                           'depth': depth, 
                                                           'use_residual': use_residual, 
                                                           'bottleneck_size': bottleneck_size, 
                                                           'number_of_filters': number_of_filters})

hp_tuning_config = { 
    'experiment_name': hp_tuning_experiment_name, 
    'max_num_classifiers': hp_tuning_max_num_classifiers,
    'gs_model_configurations': hp_tuning_model_configurations
}

# NUM CLASSIFIERS ANALYSIS

hp_num_classifiers_experiment_name = 'hp_num_classifiers_results' 

hp_num_classifiers_max_num_classifiers = 32

hp_num_classifiers_model_configurations = []

for kernel_sizes in [[20, 30, 40]]:

    for depth in [3]:

        for use_residual in [False]: 

            for bottleneck_size in [32]:
                
                for number_of_filters in [8]:

                    hp_num_classifiers_model_configurations.append({'kernel_sizes': kernel_sizes, 
                                                                    'depth': depth, 
                                                                    'use_residual': use_residual, 
                                                                    'bottleneck_size': bottleneck_size,
                                                                    'number_of_filters': number_of_filters})

hp_num_classifiers_config = { 
    'experiment_name': hp_num_classifiers_experiment_name, 
    'max_num_classifiers': hp_num_classifiers_max_num_classifiers,
    'gs_model_configurations': hp_num_classifiers_model_configurations
}

# BOTTLENECK SIZE ANALYSIS

hp_bottleneck_size_experiment_name = 'hp_bottleneck_size_results' 

hp_bottleneck_size_min_num_classifiers = 3

hp_bottleneck_size_max_num_classifiers = 3

hp_bottleneck_size_model_configurations = []

for kernel_sizes in [[20, 30, 40]]:

    for depth in [3]:

        for use_residual in [False]: 

            for bottleneck_size in [2, 4, 8, 16, 32, None]:
                
                for number_of_filters in [8]:

                    hp_bottleneck_size_model_configurations.append({'kernel_sizes': kernel_sizes, 
                                                                    'depth': depth, 
                                                                    'use_residual': use_residual, 
                                                                    'bottleneck_size': bottleneck_size, 
                                                                    'number_of_filters': number_of_filters})

hp_bottleneck_size_config = { 
    'experiment_name': hp_bottleneck_size_experiment_name, 
    'min_num_classifiers': hp_bottleneck_size_min_num_classifiers,
    'max_num_classifiers': hp_bottleneck_size_max_num_classifiers,
    'gs_model_configurations': hp_bottleneck_size_model_configurations
}

# RESIDUAL CONNECTIONS ANALYSIS

hp_residual_experiment_name = 'hp_residual_results' 

hp_residual_min_num_classifiers = 3

hp_residual_max_num_classifiers = 3

hp_residual_model_configurations = []

for kernel_sizes in [[20, 30, 40]]:

    for depth in [3]:

        for use_residual in [False, True]:

            for bottleneck_size in [32]:
                
                for number_of_filters in [8]:

                    hp_residual_model_configurations.append({'kernel_sizes': kernel_sizes, 
                                                             'depth': depth, 
                                                             'use_residual': use_residual, 
                                                             'bottleneck_size': bottleneck_size, 
                                                             'number_of_filters': number_of_filters})

hp_residual_config = { 
    'experiment_name': hp_residual_experiment_name, 
    'min_num_classifiers': hp_residual_min_num_classifiers,
    'max_num_classifiers': hp_residual_max_num_classifiers,
    'gs_model_configurations': hp_residual_model_configurations
}

# NETWORK DEPTH ANALYSIS

hp_depth_experiment_name = 'hp_depth_results' 

hp_depth_min_num_classifiers = 3

hp_depth_max_num_classifiers = 3

hp_depth_model_configurations = []

for kernel_sizes in [[20, 30, 40]]:

    for depth in [1, 2, 3, 4, 5, 6]:

        for use_residual in [False]: 

            for bottleneck_size in [32]:
                
                for number_of_filters in [8]:

                    hp_depth_model_configurations.append({'kernel_sizes': kernel_sizes, 
                                                          'depth': depth, 
                                                          'use_residual': use_residual, 
                                                          'bottleneck_size': bottleneck_size, 
                                                          'number_of_filters': number_of_filters})

hp_depth_config = { 
    'experiment_name': hp_depth_experiment_name, 
    'min_num_classifiers': hp_depth_min_num_classifiers,
    'max_num_classifiers': hp_depth_max_num_classifiers,
    'gs_model_configurations': hp_depth_model_configurations
}

# NUM FILTERS ANALYSIS

hp_number_of_filters_experiment_name = 'hp_number_of_filters_results' 

hp_number_of_filters_min_num_classifiers = 3

hp_number_of_filters_max_num_classifiers = 3

hp_number_of_filters_model_configurations = []

for kernel_sizes in [[20, 30, 40]]:

    for depth in [3]:

        for use_residual in [False]: 

            for bottleneck_size in [32]:
                
                for number_of_filters in [4, 8, 16, 32]:

                    hp_number_of_filters_model_configurations.append({'kernel_sizes': kernel_sizes, 
                                                                      'depth': depth, 
                                                                      'use_residual': use_residual, 
                                                                      'bottleneck_size': bottleneck_size, 
                                                                      'number_of_filters': number_of_filters})

hp_number_of_filters_config = { 
    'experiment_name': hp_number_of_filters_experiment_name, 
    'min_num_classifiers': hp_number_of_filters_min_num_classifiers,
    'max_num_classifiers': hp_number_of_filters_max_num_classifiers,
    'gs_model_configurations': hp_number_of_filters_model_configurations
}

# FILTER LENGTH ANALYSIS

hp_filter_length_experiment_name = 'hp_filter_length_results' 

hp_filter_length_min_num_classifiers = 3

hp_filter_length_max_num_classifiers = 3

hp_filter_length_model_configurations = []

for kernel_sizes in [[8], [16], [32], [64]]:

    for depth in [3]:

        for use_residual in [False]: 

            for bottleneck_size in [32]:
                
                for number_of_filters in [8]:

                    hp_filter_length_model_configurations.append({'kernel_sizes': kernel_sizes, 
                                                                  'depth': depth, 
                                                                  'use_residual': use_residual, 
                                                                  'bottleneck_size': bottleneck_size, 
                                                                  'number_of_filters': number_of_filters})

hp_filter_length_config = { 
    'experiment_name': hp_filter_length_experiment_name, 
    'min_num_classifiers': hp_filter_length_min_num_classifiers,
    'max_num_classifiers': hp_filter_length_max_num_classifiers,
    'gs_model_configurations': hp_filter_length_model_configurations
}
