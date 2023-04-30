# Importing libraries
from hyperopt import fmin, tpe,Trials,hp,STATUS_OK, SparkTrials
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # To suppress tensorflow warnings
import tensorflow as tf
import keras
import pandas as pd
from sklearn.utils import shuffle
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np
import pickle
import sys
from gc import callbacks
import pandas as pd
import random
from tqdm import tqdm
import multiprocessing as mp
import time
import statistics
from tensorflow.keras.callbacks import EarlyStopping
from HPO import Aritificial_NN_PO, data_import_and_shuffle, Bayesian_optimization_search
from evaluation import best_models

def read_text_file(path:str):
    """
    This method is used to extract features and labels from text file, along with threshold values for evaluation

    path: directory for input text file
    output:
        features: list of features
        labels: list of labels
        percentage_thresholds: list of percentage threshold values for labels
        abs_thresholds: list of absolute threshold values for labels 
    """
    file = open(path, 'r')
    lines = file.readlines()
    # Features and labels
    features = lines[2].replace(" ", "")
    features = features[10:-2].split(',')
    labels = lines[3].replace(" ", "")
    labels = labels[8:-2].split(',')

    # Evaluation thresholds
    percentage_thresholds = lines[7]. replace(" ", "")
    percentage_thresholds = percentage_thresholds[23:-2].split(',')
    percentage_thresholds = [float(values) for values in percentage_thresholds]
    abs_thresholds = lines[8]. replace(" ", "")
    abs_thresholds = abs_thresholds[21:-2].split(',')
    abs_thresholds = [float(values) for values in abs_thresholds]
    return features, labels, percentage_thresholds, abs_thresholds

def convertion_to_int(data:pd.core.series.Series):
    """
    This method converts the number of neurons, originally imported as str, into a list of integers.
    """
    for i in range(len(data)):
        data[i]['Number of neurons'] = data[i]['Number of neurons'][1:-1].split(',')            # Spliting values at ','
        for j in range(len(data[i][2])):                    # Converting to int
            data[i]['Number of neurons'][j] = int(data[i]['Number of neurons'][j])
    return data
    
# Data preprocessing
def data_preprocessing(TCAD_data:pd.core.frame.DataFrame, evaluation_data:pd.core.frame.DataFrame, axial_data:pd.core.frame.DataFrame, no_training_data:int ,features: list,labels: list):
	"""
	This method preprocesses the data and generates scaled training and testing datasets 
	"""
	# Setting and saving random seed
	seed = random.randint(1,2000)
	tf.random.set_seed(seed)
	# Removing evaluation data from data ------> Training data, which is further narrowed down to the number of training samples.
	# Removing duplicates by 'ctrlstr' column
	training_data = TCAD_data.copy()
	duplicate_index = TCAD_data['ctrlstr'].isin(evaluation_data['ctrlstr'])
	training_data.drop(training_data[duplicate_index].index, inplace = True)
    # Finding target value for relative error calculations
	for i in range(len(axial_data)):
		if axial_data['ctrlstr'].iloc[i] == 'axes_100045':
			target_index = i
	X = pd.concat([axial_data[features],training_data[features]], ignore_index=True, sort=False)
	#X = self.data[self.features]
	Y = pd.concat([axial_data[labels],training_data[labels]], ignore_index=True, sort=False)
	# Splitting data into training and testing data
	#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state =28)
	X_train = pd.concat([axial_data[features],training_data.iloc[0:no_training_data][features]], ignore_index=True, sort=False)
	Y_train = pd.concat([axial_data[labels],training_data.iloc[0:no_training_data][labels]], ignore_index=True, sort=False)
	X_test = evaluation_data[features]
	Y_test = evaluation_data[labels]
	# Scaling the input data using Standard Scaler
	sc = StandardScaler()
	sc_output = StandardScaler()
	X_train = sc.fit_transform(X_train.values)
	X_test = sc.transform(X_test.values)
	Y_train = sc_output.fit_transform(Y_train.values)
	Y_test = sc_output.transform(Y_test.values)
	# We dont apply fit method to the test data so that the model does not compute mean standard deviation
	# of the test dataset during training
	return seed, training_data, X_train, X_test, Y_train, Y_test, sc, sc_output, target_index

# Pick best model from wide search (random) to input into narrow search (Bayesian opt.)
def top_k_pick(models_path):
    """
    This method picks the top 20 configurations from the wide search (with least MAE scores) and then looks into
    these configurations to find best configurations with varying hidden layers.
    """
    data = pd.read_csv(models_path)
    # Sorting w.r.t MAE scores in ascending order
    data = data.sort_values('Norm_MAE_avg')
    # Looking into top 20 results and selecting  k hidden layer numbers from these 20 top performing configurations 
    top_k = list(set(data['Number of hidden layers'].iloc[0:20]))    # Converting to set first to remove duplicates
    # Picking configurations with lowest MAE score for hidden layer numbers selected from top_20 (lowest MAE configuration for each of the selected HL numbers)
    top_k_data = []
    for top_values_index in range(len(top_k)):
        for data_index in range(len(data)):
            if data['Number of hidden layers'].iloc[data_index] == top_k[top_values_index]:
                top_k_data.append(data.iloc[data_index])
                break
    top_k_data = convertion_to_int(top_k_data)
    # Extracting number of neurons from top selected configurations. 
    top_config_no_neurons = []
    for configurations in top_k_data:
        top_config_no_neurons.append(configurations['Number of neurons'])
    return top_config_no_neurons

if __name__ == "__main__":
    # Setting the parameters from shell script
    cpu_number = int(sys.argv[1])
    no_training_data = int(sys.argv[2])
    min_no_hl, max_no_hl = int(sys.argv[3]), int(sys.argv[4])
    min_no_neurons,max_no_neurons = int(sys.argv[5]), int(sys.argv[6])
    activation_func = sys.argv[7]
    epochs, training_repetitions = int(sys.argv[8]), int(sys.argv[9])
    random_search = sys.argv[10]
    configurations = int(sys.argv[11])
    narrow_search = sys.argv[12]
    max_evals = int(sys.argv[13])         # For narrow search
    input_path = sys.argv[14]
    evaluation_path = sys.argv[15]
    axial_input = sys.argv[16]
    output_path = sys.argv[17]
    threshold_eval = sys.argv[18]
    eval_type = sys.argv[19]
    eval_data_path = sys.argv[20]

    # Importing TCAD data
    TCAD_data = data_import_and_shuffle(input_path)
    evaluation_data = data_import_and_shuffle(evaluation_path)
    axial_data = data_import_and_shuffle(axial_input)
    features, labels, percentage_thresholds, abs_thresholds = read_text_file(eval_data_path)
    seed, training_data, X_train, X_test, Y_train, Y_test, sc, sc_output, target_index = data_preprocessing(TCAD_data, evaluation_data, axial_data, no_training_data, features, labels)
    # Parallelization
    pool = mp.Pool(mp.cpu_count())
    
    # Wide space search (using random search)
    print("Initiating wide range random search")
    if random_search == 'True':
        nn = Aritificial_NN_PO(training_data, axial_data, X_train, X_test, Y_train, Y_test, target_index, sc, sc_output, min_no_hl, max_no_hl, min_no_neurons, max_no_neurons, features, labels, activation_func, epochs ,training_repetitions, output_path,evaluation_data)
        # pool.map(nn.generate_output, [1 for _ in range(configurations)])
        nn.generate_output(configurations)
    
    # Selecting the top results from wide search (Input for narrow search)
    print("Initiating narrow range Bayesian search")
    if narrow_search == 'True':
        top_config_neurons = top_k_pick(output_path)
        # Narrow search (Bayesian Optimization search) on each of the top configs selected from wide search, also parallelized
        bayesian_search = Bayesian_optimization_search(training_data, axial_data, X_train, X_test, Y_train, Y_test, target_index, sc, sc_output, seed, output_path, features, labels, activation_func, epochs, max_evals)
        pool.map(bayesian_search.bayesian_method_implementation, [top_config for top_config in top_config_neurons])
   
    # Threshold Evaluation 
    print("Initiating threshold evaluation")
    if threshold_eval == 'True':
        best_models(pd.read_csv(output_path), labels, eval_type= eval_type, percentage_threshold= percentage_thresholds, abs_threshold= abs_thresholds)
