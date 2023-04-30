# Importing Libraries
from gc import callbacks
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # To suppress tensorflow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras
from hyperopt import fmin, tpe,Trials,hp,STATUS_OK, SparkTrials
import json
import random
from tqdm import tqdm
import multiprocessing as mp
import time
import sys
import statistics
import pickle
from tensorflow.keras.callbacks import EarlyStopping
import csv

# Importing data

def data_import_and_shuffle(filepath:str):
    """
    parameter: filepath which is the directory of the datafile.

    This method imports the data file, removes the first column which is index column
    and returns the shuffles the data
    """
    data = pd.read_csv(filepath)
    data = data.dropna()
    if ('Unnamed: 0' in data):			# Remove unnecessary data
        data = shuffle(data.drop(columns = ['Unnamed: 0']))
    return data

def output_label_gen(labels):
    """
    parameter: labels/ electrical parameter names

    This method generates a set of labels for each electrical parameter. One representing percentage scores
    and the second representing absolute differences between true and predicted label values
    """
    labels_for_out = []
    for label in labels:
        labels_for_out.append(label+"_Res%")       # To represent percentage value
        labels_for_out.append(label+"_abs")     # To represent absolute 
        # labels_for_out.append(label+"_std")     # To represent std of error
    return labels_for_out

def output_score_format(scores):
    """
    parameter: Percentage and absoulte difference scores for each label

    This method is used to set the format/order of the scores to a more presentable one.
    e.g : Output ---> current_%, current_abs, voltage_%, voltage_abs,....
    """
    percentage_score = scores[0]
    abs_score = scores[1]
    output_scores = []
    for i in range(len(percentage_score)):
        output_scores.append(percentage_score[i])
        output_scores.append(abs_score[i])
        # output_scores.append(error_std[i])
    return output_scores
        
def output_to_excel(list_of_data:list, output_path:str):
    """
    parameter: 
        A list of lists containing NN strucutre data (Number of hidden layers, Number of neurons, MAE values and runtime information)
        output path for the excel sheet containing model score and info.
    
    This method creates a dataframe based on the output list from the class and exports it into .csv file (Excel sheet) 
    """
    output_labels = output_label_gen(list_of_data[3])
    output_scores = output_score_format(list_of_data[2])
    df = pd.DataFrame()
    df["Number of hidden layers"] = list_of_data[0]
    df["Number of neurons"] = list_of_data[1]
    df["Norm_MAE_avg"] = list_of_data[2][2]
    for i in range(len(output_labels)):        # For total number of scores
        df[output_labels[i]] = output_scores[i]
    df["Average runtime per training (min)"] = list_of_data[4]
    # Saving the dataframe to an excel file
    output= output_path
    df.to_csv(output_path, mode='a', header=not os.path.exists(output))		# This will append the datafile and put headers only for the first time.

def convertion_to_float(data:pd.core.series.Series):
    """
    This method converts number of neurons, that are being imported as a string, into a list.
    """
    data = data.apply(lambda x: x[1:-1].split(','))
    for values in data:
        for i in range(len(values)):
            values[i] = float(values[i])
    return data
###################################################################################################################

# Neural Network Class (For Random Search)
class Aritificial_NN_PO:
    def __init__(self, training_data:pd.core.frame.DataFrame, axial_data: pd.core.frame.DataFrame, X_train:pd.core.frame.DataFrame, X_test:pd.core.frame.DataFrame,
    Y_train:pd.core.frame.DataFrame, Y_test:pd.core.frame.DataFrame, target_index: int, sc, sc_output, min_no_hl:int, max_no_hl: int, min_no_neurons:int, max_no_neurons: int, 
    features: list,labels:list,activation_func :str, epochs:int, training_repitition:int, output_path:str, evaluation_data:pd.core.frame.DataFrame) -> None:
        """
        Constructor method for the class which has parameters:
        training_data: TCAD training dataset
        axial_data: Dataframe with axial and 2D feature corner samples
        X_train: TCAD feature training dataset
        X_test: TCAD feature testing dataset
        Y_train: TCAD labels/electrical parameters training dataset
        Y_test: TCAD labels/electrical parameters testing dataset
        target_index: index of TCAD simulation with target values set for features
        sc: standard scaler used for transforming TCAD features
        sc_output: standard scaler used for transforming TCAD labels
        min_no_hl: lower limit for range of number of hidden layers for the neural network structure
        max_no_hl: Upper limit for range of number of hidden layers for the neural network structure
        min_no_neurons: lower limit for range of number of neurons per hidden layer for the neural network structure
        max_no_neurons: Upper limit for range of number of neurons per hidden layer for the neural network structure
        features: list of features in TCAD data
        labels: list of labels in TCAD data
        activation_func: Activation function to use during neural network's training
        epochs: Defines the number of times NN model will go through the entire dataset during the training process.
        training_repitition: defines how many times model is trained on the selected configuration of the neural network's structure.
        output_path: output directory path for appending trained Neural network configurations along with their scores.
        """
        self.training_data = training_data
        self.axial_data = axial_data
        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test 
        self.target_index = target_index
        self.sc, self.sc_output = sc, sc_output
        self.min_no_hl = min_no_hl
        self.max_no_hl = max_no_hl
        self.min_no_neurons = min_no_neurons
        self.max_no_neurons = max_no_neurons
        self.features = features
        self.labels = labels
        self.activation_func = activation_func
        self.epochs = epochs
        self.training_repitition = training_repitition
        self.output_path = output_path
        self.evaluation_data = evaluation_data

    def neural_network_structure(self):
        """
        This method sets the structure of the neural network based on the hyperparameters defined,
        it sets up the hidden layers with the certain number of neurons.
        The range for selecting number of hidden layers and number of neurons is set by the hyperparameters which
        define the lower and upper limits.
        """
        # Initializing the Neural Network
        nn = tf.keras.models.Sequential()
        # Adding the input layer
        nn.add(tf.keras.layers.Input(shape=(self.X_train.shape[1],)))
        # Adding hidden layers based on random selection between the defined range
        no_hidden_layers = random.randint(self.min_no_hl,self.max_no_hl)
        neurons = []
        for _ in range(1, no_hidden_layers): 
            # Using random range with a step to get values divisble by a step size of 5
            no_of_neurons = random.randrange(self.min_no_neurons,self.max_no_neurons,5)
            # Saving the neurons selected in a list
            neurons.append(no_of_neurons)
            nn.add(tf.keras.layers.Dense(units =no_of_neurons , activation = self.activation_func))        
        # Adding the output layer
        nn.add(tf.keras.layers.Dense(units=self.Y_train.shape[1], activation = 'linear'))
         # Compiling the model
        # mae_percent = 'MeanAbsolutePercentageError'
        nn.compile(optimizer='Adam', loss = 'mse', metrics = 'mae')
        """
        using MAE from keras library is justifiable here because all the labels have been scaled
        using standard scaler and thus the values exist in a similar range for all the labels.
        Therefore it is not needed to evaluate the loss seperately for each label. 
        """         
        initial_weights = nn.get_weights()                               # Saving initial weights
        return nn,no_hidden_layers-1, neurons, initial_weights           # -1 because the upper limit is not included in range method
    
    def training_neural_network(self, nn, initial_weights):
        """
        This method runs the fit method of neural network and trains the model on the dataset.
        It is reset everytime with initial weights of the model in order to ensure learning from scratch
        parameters:
        nn: neural network with the defined structure defined in neural_network_structure method
        initial_weights: These weights are the initial weights on which the model is compiled.
        """
        # Adding early stopping
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
        # Loading initial weights to reset the training of neural network
        nn.set_weights(initial_weights)
        # Setting custom weights, to increase training prirority for axial samples.
        custom_weights = {}
        a = 0.5			# For normalizing weights
        for i in range(len(self.axial_data) + len(self.training_data)):
            if (i <= len(self.axial_data)):
                custom_weights[i] = a/len(self.axial_data)
            else:
                custom_weights[i] = (1-a)/len(self.training_data)
		# Training NN with adapted weights and early stopping.
        nn.fit(self.X_train, self.Y_train, batch_size=32, epochs = self.epochs, class_weight = custom_weights ,verbose = 0, validation_split = 0.2,callbacks = [early_stopping])
        print('Model Trained')
        return nn

    def error_per_label(self,pred_values):
        """
        This method calculates mean absolute error scores per label and returns a list
        parmaters:
        true_values: These are the actual label values stored in the test set on which the model has not been trained
        pred_values: These are the predicted label values attained from the model after training completion
        """
        # Scaling the data back to its original representation.
        true_values = self.sc_output.inverse_transform(self.Y_test)
        pred_values_transform = self.sc_output.inverse_transform(pred_values)
        target_value_transform = self.sc_output.inverse_transform(self.Y_train)
        percentage_error = []
        absolute_error = []
        print('Looking for bad simulations')
        for i in range(true_values.shape[1]):   # Number of labels
            failed_sim = pd.DataFrame()
            percentage_error_per_label = []
            absolute_error_per_label = []
            for row_number in range(true_values.shape[0]):
                percentage_error_per_label.append(abs(true_values[row_number][i] - pred_values_transform[row_number][i])/(target_value_transform[self.target_index][i])*100)
                absolute_error_per_label.append(abs(true_values[row_number][i] - pred_values_transform[row_number][i]))
            if i == self.labels.index('Idsat_sv') or i == self.labels.index('Idlin_sv') or i == self.labels.index('Ideff_sv') or i == self.labels.index('Vtsat_sv') or i == self.labels.index('Vtlin_sv') or i == self.labels.index('DIBL_sv'):
                for j in range(len(percentage_error_per_label)):
                    if percentage_error_per_label[j] >= 5:            # Checking for simulations failing with >= 10% difference per label.
                        failed_sim = pd.concat([failed_sim, self.evaluation_data.iloc[j].to_frame().T], ignore_index=True)      # evaluation data and test data have same indexing
                failed_sim.drop_duplicates()
                if os.path.isdir('Failed_simulations') == False:
                    os.mkdir('Failed_simulations')
                if os.path.isdir('Failed_simulations/{}'.format(self.labels[i])) == False:
                    os.mkdir('Failed_simulations/{}'.format(self.labels[i]))
                output = 'Failed_simulations/{}/failed_simulations.csv'.format(self.labels[i])   
                failed_sim.to_csv(output, mode ='a', header=not os.path.exists(output), index = False)
            percentage_error.append(max(percentage_error_per_label))
            absolute_error.append(max(absolute_error_per_label))
        return [percentage_error,absolute_error]
    
    def evaluate_model(self,nn):
        """
        This method prints the loss (mean absolute error) computed from the trained model.
        parameter:
        nn: trained neural network obtained from training_neural_network method
        """
        predicted_values = nn.predict(self.X_test)
        score = self.error_per_label(predicted_values)
        loss = nn.evaluate(self.X_test ,self.Y_test, verbose=0)
        score.append(loss[1])         # To retur percentage_error_score per label and overall MAE_score 
        return score      
    
    def check_for_repitition (self, neurons:list) -> list:
        """
        This method prevents repitition of a neural network structure with same hidden layers and neurons (Because
        we want to explore the configuration space widely). Therefore, if there is a repition, then this method
        will repeat the formation of neural network structure untill there is no repitiion.
        """
        filepath = self.output_path
        if os.path.exists(filepath):
            output_file = pd.read_csv(filepath)
            output_file = output_file.dropna()
            total_neurons = convertion_to_float(output_file['Number of neurons'])
            for previous_neurons in total_neurons:
                if neurons != previous_neurons:
                    pass
                else:
                    print('\n\n Repitition Allert!!! \n\n')
                    # if there is repitition, then we call the structure method again to create a new neural network structure.
                    _,_,neurons,_ = self.neural_network_structure()
                    # _ ignores the output (as we do not want the neural network and number of hidden layers again)
                    _,neurons = self.check_for_repitition(neurons)
        return len(neurons), neurons        # Length of neurons  = number of hidden layers
    
    def generate_output(self, configurations:int):
        """
        parameter: configurations decides the number of times NN runs with different number of hidden layers and number of neurons
        This method generates the output file which contains:
        number of hidden layers, neurons in each of those layers and mean absolute error values per training
        """ 
        for _ in tqdm(range(configurations)):
            untrained_nn, no_hidden_layers, neurons, initial_weights = self.neural_network_structure()
            # Check for repitition
            no_hidden_layers, neurons = self.check_for_repitition(neurons)
            for i in range(0, self.training_repitition):
                 # Start time
                st = time.time()
                nn = self.training_neural_network(untrained_nn, initial_weights)
                loss = self.evaluate_model(nn)
                # End time
                et = time.time()
                # Elapsed time per training will simply be end time - start time divided by the number of training repititions
                # in order to get average runtime for a single training
                runtime = (et-st)/60                     # In minutes
                output_to_excel([[no_hidden_layers], [neurons], loss, self.labels, runtime], self.output_path)
###################################################################################################################

# Bayesian Optimization search (Narrow search)
class Bayesian_optimization_search:
    def __init__(self, training_data:pd.core.frame.DataFrame, axial_data:pd.core.frame.DataFrame, X_train:pd.core.frame.DataFrame, X_test:pd.core.frame.DataFrame,
        Y_train:pd.core.frame.DataFrame, Y_test:pd.core.frame.DataFrame, target_index:int, sc, sc_output, seed, output_path:str, features: list, labels: list, activation_func: str,
        epochs: int, max_evals: int) -> None:
        """
        Constructor method for the class which has parameters:
        training_data: TCAD training dataset
        axial_data: Dataframe with axial and 2D feature corner samples
        X_train: TCAD feature training dataset
        X_test: TCAD feature testing dataset
        Y_train: TCAD labels/electrical parameters training dataset
        Y_test: TCAD labels/electrical parameters testing dataset
        target_index: index of TCAD simulation with target values set for features
        sc: standard scaler used for transforming TCAD features
        sc_output: standard scaler used for transforming TCAD labels
        seed: Tensorflow random seed stored in order to recreate data samples and NN model at any point.
        output_path: output directory path for appending trained Neural network configurations along with their scores.
        features: list of features in TCAD data
        labels: list of labels in TCAD data
        activation_func: Activation function to use during neural network's training
        epochs: Defines the number of times NN model will go through the entire dataset during the training process.
        max_evals: Maxmimum number of bayesian optimization evaluations per NN configuration.
        """
        self.training_data = training_data
        self.axial_data = axial_data
        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test
        self.target_index = target_index
        self.sc, self.sc_output = sc, sc_output
        self.seed = seed
        self.output_path = output_path
        self.features = features
        self.labels = labels
        self.activation_func = activation_func
        self.epochs = epochs
        self.max_evals = max_evals    

    def save_model(self,nn,model_name):
        """
        This method saves the model in a specified directory
        parameter:  
        model_name: This method saves the model with the parameter 'model_name'
        """
        model_json = nn.to_json()
        # Check if directory exists
        if os.path.isdir('saved_model') == False:
            os.mkdir('saved_model')
        if os.path.isdir("saved_model/{}".format(model_name)) == False:
            os.mkdir("saved_model/{}".format(model_name))   
        # Saving seed
        with open('saved_model/{}/seed.txt'.format(model_name), 'w') as f:
            f.write('Seed : {}'.format(self.seed))    
		# Saving model to JSON
        with open("saved_model/{}/{}.json".format(model_name, model_name), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        nn.save_weights("saved_model/{}/{}.h5".format(model_name,model_name),"w")
        # Saving scalers
        pickle.dump(self.sc, open('saved_model/{}/scaler_feautures.pkl'.format(model_name), 'wb'))
        pickle.dump(self.sc_output, open('saved_model/{}/scaler_labels.pkl'.format(model_name), 'wb'))    

    def error_per_label(self,pred_values):
        """
        This method calculates mean absolute error scores per label and returns a list
        parmaters:
        true_values: These are the actual label values stored in the test set on which the model has not been trained
        pred_values: These are the predicted label values attained from the model after training completion
        """
        # Scaling the data back to its original representation.
        true_values = self.sc_output.inverse_transform(self.Y_test)
        pred_values_transform = self.sc_output.inverse_transform(pred_values)
        target_value_transform = self.sc_output.inverse_transform(self.Y_train)
        percentage_error = []
        absolute_error = []
        for i in range(true_values.shape[1]):   # Number of labels
            percentage_error_per_label = []
            absolute_error_per_label = []
            for row_number in range(true_values.shape[0]):
                # if true_values[row_number][i] != 0 and pred_values_transform[row_number][i] != 0:
                # percentage_error_per_label.append(abs(true_values[row_number][i] - pred_values_transform[row_number][i])/((true_values[row_number][i] + pred_values_transform[row_number][i])/2)*100)
                percentage_error_per_label.append(abs(true_values[row_number][i] - pred_values_transform[row_number][i])/(target_value_transform[self.target_index][i])*100)
                absolute_error_per_label.append(abs(true_values[row_number][i] - pred_values_transform[row_number][i]))
            percentage_error.append(max(percentage_error_per_label))
            absolute_error.append(max(absolute_error_per_label))
        return [percentage_error,absolute_error]

    def evaluate_model(self,nn):
        """
        This method prints the loss (mean absolute error) computed from the trained model.
        parameter:
        nn: trained neural network obtained from training_neural_network method
        """
        predicted_values = nn.predict(self.X_test)
        score = self.error_per_label(predicted_values)
        loss = nn.evaluate(self.X_test ,self.Y_test, verbose=0)
        score.append(loss[1])         # To retur percentage_error_score per label and overall MAE_score 
        return score   

    # Defining objective function for Bayesian opt.
    def objective_func(self, params):
        """
        This method, which is our NN implementation, serves as the objective function for Bayesian optimization
        and outputs number of neurons in each hidden layers along with MAE score.
        """
        # Start time
        st = time.time()
        # no_neurons_hl_1, no_neurons_hl_2, no_neurons_hl_3, no_neurons_hl_4 = args
        nn = keras.models.Sequential()
        # Adding input layer
        nn.add(keras.layers.Input(shape = (self.X_train.shape[1],)))
        # Counting number of hidden layers
        no_hidden_layers = 0
        for i in range(len(self.number_of_neurons)):
            # Adding hidden layers
            nn.add(keras.layers.Dense(units = params['no_neurons_hl_{}'.format(i+1)], activation = self.activation_func))
            no_hidden_layers+=1
        # Adding output layer
        nn.add(keras.layers.Dense(units = self.Y_train.shape[1], activation = 'linear'))
        nn.compile(optimizer = 'Adam', loss = 'mse', metrics = 'mae')
        # Adding early stopping
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20)
        # Setting custom weights, to increase training prirority for axial samples.
        custom_weights = {}
        a = 0.5			# For normalizing weights
        for i in range(len(self.axial_data) + len(self.training_data)):
             if (i <= len(self.axial_data)):
                custom_weights[i] = a/len(self.axial_data)
             else:
                custom_weights[i] = (1-a)/len(self.training_data)
		# Training NN with adapted weights and early stopping.
        nn.fit(self.X_train, self.Y_train, batch_size=32, epochs = self.epochs, class_weight = custom_weights ,verbose = 0, validation_split = 0.2,callbacks = [early_stopping])
        # loss = nn.evaluate(self.X_test,self.Y_test, verbose = 0)
        loss = self.evaluate_model(nn)
        # End time
        et = time.time()
        # Elapsed time per training will simply be end time - start time divided by the number of training repititions
        # in order to get average runtime for a single training
        runtime = (et-st)/60                     # In minutes
        output_to_excel([[no_hidden_layers], [self.number_of_neurons], loss, self.labels, runtime], self.output_path)
        # Saving model and its weights
        # Using MAE_score as model name for saving purpose. (MAE score will be unique for each configuration)
        model_name = 'model_with_MAE_' + str(loss[-1])
        self.save_model(nn,model_name)
        # Adding parameter ,loss and status values to output dictionary
        parameters = {}
        for i in range(len(self.number_of_neurons)):
            parameters['no_neurons_hl_{}'.format(i+1)] = params['no_neurons_hl_{}'.format(i+1)]
        output_dict = {}
        output_dict['params'] = parameters
        output_dict['loss'] = loss[-1]
        output_dict['status'] = STATUS_OK
        return output_dict

    def search_space(self):
        """
        This method defines the search space, ranges for number of neurons in each hidden layers,
        for the bayesian method to choose from and optimize.
        """
        neuron_range = 10           # This variable is used to set the range for searching optimized values for number of neurons.
        search_space = {}
        for i in range(len(self.number_of_neurons)):
            search_space['no_neurons_hl_{}'.format(i+1)] = hp.quniform('no_neurons_hl_{}'.format(i+1),self.number_of_neurons[i]-neuron_range,self.number_of_neurons[i]+neuron_range,2)
        return search_space

    def bayesian_method_implementation(self, number_of_neurons:list):
        """
        This method implements bayesian optimization search (narrow search), and finds the best configuration
        for the given neural network and dataset.
        Parameter:
        number_of_neurons: Number of neurons from top configurations found by wide search (Using random search)
        """
        # Setting number of neurons for the class, to be utilized by other methods with in the class.
        self.number_of_neurons = number_of_neurons
        # For parallelelizm
        # trials = SparkTrials(parallelism=2)          # Distribute tuning across Spark cluster
        trials = Trials()
        # func = self.objective_func(number_of_neurons = number_of_neurons)
        best_hp = fmin(fn = self.objective_func,
            space = self.search_space(),
            algo = tpe.suggest,     # Tree of Parzen Estimators, a Bayesian method
            max_evals= self.max_evals,
            trials = trials)
        best_configs = json.dumps(best_hp)
        results = trials.results
        loss = []
        for _, r in enumerate(results):
            loss.append(r['loss'])
        mae_score = min(loss)
        # appending best config info along with MAE score into a .txt file
        with open('best_configs.txt', 'a') as f:
            f.write(best_configs)
            f.write(', MAE_score : {}'.format(mae_score))
            f.write("\n")
