# POOLING ANALYSIS (FAP vs GAP)

pooling_experiment_name = 'pooling_benchmark'

pooling_min_num_classifiers = 3

pooling_max_num_classifiers = 3

pooling_model_configurations = []

for kernel_sizes in [[20, 30, 40]]:

    for depth in [3]:

        for use_residual in [False]: 

            for bottleneck_size in [32]:
                
                for number_of_filters in [8]:
                
                    for global_pool in [False, True]:

                        pooling_model_configurations.append({'kernel_sizes': kernel_sizes, 
                                                             'depth': depth, 
                                                             'use_residual': use_residual, 
                                                             'bottleneck_size': bottleneck_size, 
                                                             'number_of_filters': number_of_filters, 
                                                             'global_pool': global_pool})

pooling_benchmark_config = { 
    'experiment_name': pooling_experiment_name, 
    'min_num_classifiers': pooling_min_num_classifiers,
    'max_num_classifiers': pooling_max_num_classifiers,
    'gs_model_configurations': pooling_model_configurations
}



