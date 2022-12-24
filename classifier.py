import keras
import numpy as np
import tensorflow as tf
from keras import Model
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.layers import Activation, Add, Input, Conv1D, Flatten, Dense, AvgPool1D, MaxPool1D, Concatenate, BatchNormalization, Dropout
from keras.layers import GlobalAveragePooling1D
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.model_selection import train_test_split, KFold


MODEL_FILE_PATH_TEMPLATE = '{}_{}_best_model.hdf5'


def get_instinct_inception_module(input_tensor, kernel_sizes, stride, activation, bottleneck_size=None, number_of_filters=8):
    conv_input = input_tensor
	
    if bottleneck_size is not None and input_tensor.shape[-1] > bottleneck_size:
	
        conv_input = Conv1D(filters=bottleneck_size, 
                            kernel_size=1,
                            padding='same', 
                            activation=activation, 
                            use_bias=False)(input_tensor)
    
    num_filters = number_of_filters

    conv_list = []

    for kernel_size in kernel_sizes:
	
        conv_list.append(Conv1D(filters=num_filters, 
                                kernel_size=kernel_size,
                                strides=stride, 
                                padding='same', 
                                activation=activation, 
                                use_bias=False)(conv_input))

    max_pool = MaxPool1D(pool_size=3, 
                         strides=stride, 
                         padding='same')(input_tensor)

    additional_conv = Conv1D(filters=num_filters, 
                             kernel_size=1,
                             padding='same', 
                             activation=activation, 
                             use_bias=False)(max_pool)

    conv_list.append(additional_conv)

    convs = Concatenate(axis=2)(conv_list)

    normalized = BatchNormalization()(convs)

    output = Activation(activation='relu')(normalized)
    
    return output

	
def get_instinct_shortcut_layer(input_tensor, out_tensor, stride):
    shortcut = Conv1D(filters=int(out_tensor.shape[-1]), 
                      kernel_size=1,
                      strides=stride,
                      padding='same', 
                      use_bias=False)(input_tensor)
    
    normalized = BatchNormalization()(shortcut)

    output = Add()([normalized, out_tensor])
    
    output = Activation('relu')(output)
    
    return output
	

def build_single_instinct_model(input_shape, file_path, depth, use_residual, num_classes, kernel_sizes, stride=1, 
							    activation='linear', bottleneck_size=None, add_checkpoint=True, print_summary=False, 
							    global_pool=False, dropout=0, number_of_filters=8):
    input_layer = Input(input_shape)

    x = input_layer
    
    residual = input_layer

    for d in range(depth):
        
        layer_stride = stride if d % 2 == 0 else 1 

        x = get_instinct_inception_module(x,
									 	  kernel_sizes=kernel_sizes, 
									  	  stride=layer_stride, 
										  activation=activation, 
										  bottleneck_size=bottleneck_size, 
										  number_of_filters=number_of_filters)

        if dropout > 0:
            x = Dropout(dropout)(x)

        if use_residual and d % 2 == 1:
            x = get_instinct_shortcut_layer(residual, x, stride)
            residual = x

    if global_pool: 
        
        flattened = GlobalAveragePooling1D()(x)
    
    else:
        
        avg_pool = AvgPool1D(pool_size=3,
                             padding='same')(x)

        flattened = Flatten()(avg_pool)
    
    output_layer = Dense(num_classes, 
                         activation='softmax')(flattened)

    model = Model(inputs=input_layer, 
                  outputs=output_layer)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                                                  factor=0.5, 
                                                  patience=50, 
                                                  min_lr=0.00001)
    
    callbacks = [reduce_lr]
    
    if add_checkpoint:

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                           monitor='val_accuracy',
                                                           save_best_only=True)

        callbacks.append(model_checkpoint)
    
    if print_summary:
        
        model.summary()

    return model, callbacks

	
def single_model_predict(model_file_path, X_test):
    best_model = keras.models.load_model(model_file_path, compile=True)
    
    y_pred_probas = best_model.predict(X_test)
    
    y_pred = np.argmax(y_pred_probas, axis=-1)
    
    return y_pred, y_pred_probas
	

def get_scores(y, y_pred, y_pred_probas, num_classes):
    accuracy = accuracy_score(y, y_pred)
    
    if num_classes > 2:
        auc = roc_auc_score(y, y_pred_probas, multi_class='ovo', average='macro')
    else:
        auc = roc_auc_score(y, y_pred)
    
    return accuracy, auc
	

class SingleINSTINCTClassifier:
    
    def __init__(self, input_shape, dataset_name, depth, use_residual, num_classes, kernel_sizes, stride=1, activation='linear', 
                 bottleneck_size=None, add_checkpoint=True, model_idx=0, print_summary=False, global_pool=False, 
                 dropout=0, number_of_filters=8):
        self.dataset_name, self.model_idx = dataset_name, model_idx
        
        model, callbacks = build_single_instinct_model(input_shape, 
													   MODEL_FILE_PATH_TEMPLATE.format(dataset_name, model_idx), 
													   depth=depth, 
													   use_residual=use_residual,
													   num_classes=num_classes,
													   kernel_sizes=kernel_sizes, 
													   stride=stride, 
													   activation=activation,
													   bottleneck_size=bottleneck_size, 
													   add_checkpoint=add_checkpoint, 
													   print_summary=print_summary, 
													   global_pool=global_pool,
                                                       dropout=dropout,
													   number_of_filters=number_of_filters)
        
        self.model, self.callbacks = model, callbacks
        
    def fit(self, X_train, y_train, X_test=None, y_test=None, batch_size=None, epochs=100, verbose=True):

        hist = self.model.fit(X_train, y_train, 
                              batch_size=batch_size, 
                              epochs=epochs, 
                              verbose=verbose, 
                              validation_data=None if y_test is None else (X_test, y_test),
                              callbacks=self.callbacks)
        
        return hist

    def predict(self, X_test):
        model_file_path = MODEL_FILE_PATH_TEMPLATE.format(self.dataset_name, self.model_idx)

        y_pred, y_pred_probas = single_model_predict(model_file_path, X_test)
        
        return y_pred, y_pred_probas


class EnsembleINSTINCTClassifier:
    
    def __init__(self, num_classifiers, input_shape, dataset_name, depth, use_residual, num_classes, kernel_sizes, stride=1, 
                 activation='linear', bottleneck_size=None, add_checkpoint=True, print_summary=False, global_pool=False, 
                 dropout=0, number_of_filters=8):
        self.num_classifiers, self.num_classes = num_classifiers, num_classes
        
        self.dataset_name = dataset_name
        
        self.classifiers = []
        
        for i in range(self.num_classifiers):
            model = SingleINSTINCTClassifier(input_shape, 
											 dataset_name, 
											 depth=depth, 
											 use_residual=use_residual,
											 num_classes=num_classes,
											 kernel_sizes=kernel_sizes, 
											 stride=stride, 
											 activation=activation,
											 bottleneck_size=bottleneck_size,
											 add_checkpoint=add_checkpoint,
											 model_idx=i, 
											 print_summary=print_summary,                                             
											 global_pool=global_pool, 
											 dropout=dropout,
                                             number_of_filters=number_of_filters)
            
            self.classifiers.append(model)
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, batch_size=None, epochs=100, verbose=True):
        for i in range(self.num_classifiers):
            
            model = self.classifiers[i]
            
            print(f'***Fit model {i}***')
            
            hist = model.fit(X_train, 
                             y_train, 
                             X_test, 
                             y_test,
                             batch_size=batch_size, 
                             epochs=epochs, 
                             verbose=verbose)
            
    def predict(self, X_test, selected_classifiers=None):
        selected_classifiers = '1'*self.num_classifiers if selected_classifiers is None else selected_classifiers
        
        y_pred_probas_total, num_classifiers = np.zeros(shape=(X_test.shape[0], self.num_classes)), 0
        
        for i in range(self.num_classifiers):
            
            if selected_classifiers[i] == '1':
            
                model = self.classifiers[i]

                y_pred, y_pred_probas = model.predict(X_test)

                y_pred_probas_total += y_pred_probas
                
                num_classifiers += 1
            
        y_pred_probas_total /= num_classifiers
        
        y_pred_total = np.argmax(y_pred_probas_total, axis=-1)
        
        return y_pred_total, y_pred_probas_total
		
