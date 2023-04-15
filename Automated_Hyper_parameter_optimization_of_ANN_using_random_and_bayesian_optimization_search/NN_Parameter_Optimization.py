# Importing Libraries
from gc import callbacks
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # To suppress tensorflow initial warnings
import tensorflow as tf
import random
from tqdm import tqdm
import multiprocessing as mp
import time
import statistics
import os.path
import pickle
from tensorflow.keras.callbacks import EarlyStopping

# Importing data

def data_import_and_shuffle(filepath:str):
    """
    parameter: filepath which is the directory of the datafile.

    This method imports the data file, removes the first column which is index column
    and returns the shuffles the data
    """
    data = pd.read_csv(filepath)
    # Drop column with index values and shuffle it
    data = shuffle(data.drop(columns = ['Unnamed: 0']))
    return data

def output_to_excel(list_of_data:list):
    """
    parameter: A list of lists containing NN strucutre data (Number of hidden layers, Number of neurons, MAE values and runtime information)
    
    This creates a datafram based on the output list from the class and exports it into .csv file (Excel sheet) 
    """
    df = pd.DataFrame()
    df.insert(0, column= "Number of hidden layers", value = list_of_data[0])
    df.insert(1, column= "Number of neurons", value = list_of_data[1])
    df.insert(2, column= "MAE_avg", value = list_of_data[2][22])
    """
    Add additional columns here to store MAE results for individual outputs
    """
    df.insert(25, column= "Average runtime per training (min)", value = list_of_data[3])
    # Saving the dataframe to an excel file
    output_path = "Parameter_Optimization.csv" 
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))		# This will append the datafile and put headers only for the first time.

def models_data_in_excel(neurons,mae,id_key):
    df = pd.DataFrame()
    df.insert(0, column= "Neurons per layer", value = neurons)
    df.insert(1, column= "MAE score", value = mae[22])          # Only printing overall MAE
    # Using neurons as unique id_key, since each configuration is unique and non-repetitive
    df.insert(2, column= "Unique id_key", value = id_key)
    # Saving the dataframe to an excel file
    output_path = "saved_models/models_data.csv" 
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))		# This will append the datafile and put headers only for the first time.

def convertion_to_float(data:pd.core.series.Series):
    """
    This method converts number of neurons, that are being imported as a string, into a list.
    """
    data = data.apply(lambda x: x[1:-1].split(','))
    for values in data:
        for i in range(len(values)):
            values[i] = float(values[i])
    return data

# Neural Network Class

class Aritificial_NN_PO:
    def __init__(self, data:pd.core.frame.DataFrame, min_no_hl:int, max_no_hl: int, min_no_neurons:int, max_no_neurons: int, epochs:int, training_repitition:int) -> None:
        """
        Constructor method for the class which as parameter:
        data: TCAD_ dataset (shuffled)
        min_no_hl: lower limit for range of number of hidden layers for the neural network structure
        max_no_hl: Upper limit for range of number of hidden layers for the neural network structure
        min_no_neurons: lower limit for range of number of neurons per hidden layer for the neural network structure
        max_no_neurons: Upper limit for range of number of neurons per hidden layer for the neural network structure
        step_size_neurons: This parameter defines the step for the range for selecting number of neurons e.g: if its 5, then
        neurons selected would be divisble by 5 : 20,25,30...
        epochs: Defines the number of times NN model will go through the entire dataset during the training process.
        training_repitition: defines how many times model is trained on the selected configuration of the neural network's structure.
        """
        self.data = data
        self.min_no_hl = min_no_hl
        self.max_no_hl = max_no_hl
        self.min_no_neurons = min_no_neurons
        self.max_no_neurons = max_no_neurons
        self.epochs = epochs
        self.X_train,self.X_test,self.Y_train,self.Y_test, self.sc, self.sc_output = self.data_preprocessing()
        self.training_repitition = training_repitition

    def data_preprocessing(self):
        """
        This method preprocesses the data (applies log power 10 to Idoff values) and generates training and testing datasets 
        """
        X = self.data.iloc[:,0:9]
        Y = self.data.iloc[:,9:]
        # Splitting data into training and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state =28)
        # Scaling the input data using Standard Scaler
        sc = StandardScaler()
        sc_output = StandardScaler()
        X_train = sc.fit_transform(X_train.values)
        X_test = sc.transform(X_test.values)
        Y_train = sc_output.fit_transform(Y_train.values)
        Y_test = sc_output.transform(Y_test.values)
        # Saving scalers
        pickle.dump(sc, open('scaler_feautures.pkl', 'wb'))
        pickle.dump(sc_output, open('scaler_labels.pkl', 'wb'))
        # We dont apply fit method to the test data so that the model does not compute mean standard deviation
        # of the test dataset during training
        return X_train, X_test, Y_train, Y_test, sc, sc_output

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
        for i in range(1, no_hidden_layers): 
            # Using random range with a step to get values divisble by a step size of 5
            no_of_neurons = random.randrange(self.min_no_neurons,self.max_no_neurons,5)
            # Saving the neurons selected in a list
            neurons.append(no_of_neurons)
            nn.add(tf.keras.layers.Dense(units =no_of_neurons , activation = 'PReLU'))        
        # Adding the output layer
        nn.add(tf.keras.layers.Dense(units=self.Y_train.shape[1], activation = 'linear'))
         # Compiling the model
        # mae_percent = 'MeanAbsolutePercentageError'
        nn.compile(optimizer='Adam', loss = 'mse', metrics = ['mae'])
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
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20)
        # Loading initial weights to reset the training of neural network
        nn.set_weights(initial_weights)
        nn.fit(self.X_train, self.Y_train, batch_size=32, epochs = self.epochs, verbose = 2, validation_split = 0.2,callbacks = [early_stopping])
        return nn

    def mae_per_label(self,pred_values):
        """
        This method calculates mean absolute error scores per label and returns a list
        parmaters:
        true_values: These are the actual label values stored in the test set on which the model has not been trained
        pred_values: These are the predicted label values attained from the model after training completion
        """
        # Scaling the data back to its original representation.
        true_values = self.sc_output.inverse_transform(self.Y_test)
        pred_values_transform = self.sc_output.inverse_transform(pred_values)
        mae = []
        for i in range(true_values.shape[1]):   # Number of labels
            mean_abs_label_value = []
            for row_number in range(true_values.shape[0]):
                mean_abs_label_value.append(abs(true_values[row_number][i] - pred_values_transform[row_number][i]))
            mae.append(statistics.mean(mean_abs_label_value))
        return mae
    
    def evaluate_model(self,nn):
        """
        This method prints the loss (mean absolute error) computed from the trained model.
        parameter:
        nn: trained neural network obtained from training_neural_network method
        """
        predicted_values = nn.predict(self.X_test)
        mae = self.mae_per_label(predicted_values)
        loss = nn.evaluate(self.X_test ,self.Y_test, verbose=0)
        mae.append(loss[1])         # To retur MAE_score per label and overall MAE_score 
        return mae      
    
    def check_for_repitition (self, neurons:list) -> list:
        """
        This method prevents repitition of a neural network structure with same hidden layers and neurons (Because
        we want to explore the configuration space widely). Therefore, if there is a repition, then this method
        will repeat the formation of neural network structure untill there is no repitiion.
        """
        filepath = 'Parameter_Optimization_test.csv'
        if os.path.exists(filepath):
            output_file = pd.read_csv(filepath)
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

    def save_model(self,nn,model_name):
        """
        This method saves the model in a specified directory
        parameter:  
        model_name: This method saves the model with the parameter 'model_name'
        """
        model_json = nn.to_json()
        with open("Saved_models/{}.json".format(model_name), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        nn.save_weights("Saved_model_weights/{}.h5".format(model_name),"w")
    
    def generate_ouput(self, configurations:int):
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
                # Adding training repetition number as an additional id with neurons
                id_key = i+1
                # Creating a model name for saving purpose.
                neuron_id = '_'.join(str(values) for values in neurons)
                idkey = '__{}'.format(id_key)
                model_name = neuron_id + idkey
                self.save_model(nn,model_name)
                # Using neurons as a unique id_key to store the model. Since each configuration is unique and non repetitive,
                # we can use it as an id_key.
                models_data_in_excel([neurons],loss,model_name)
                # End time
                et = time.time()
                # Elapsed time per training will simply be end time - start time divided by the number of training repititions
                # in order to get average runtime for a single training
                runtime = (et-st)/60                     # In minutes
                output_to_excel([[no_hidden_layers],[neurons],loss,runtime])


if __name__ == "__main__":
    data = data_import_and_shuffle('filepath')
    # Setting up the configuration space.
    nn = Aritificial_NN_PO(data, min_no_hl = 3, max_no_hl = 9, min_no_neurons = 30, max_no_neurons = 55, epochs = 500 ,training_repitition = 3)
    # Start time
    st = time.time()
    # Number of configurations for random search
    configurations = 500
    # Parallelization
    pool = mp.Pool(mp.cpu_count())
    pool.map(nn.generate_ouput, [1 for _ in range(configurations)])       
    # The second parameter defines the number of iterations for each CPU.
    # So the iterator would have [1,1,1] for configurations = 3

    # Ending time
    et = time.time()
    # Calculating time to finish the process
    elapsed_time = (et - st)/60     # In minutes
    print("\n\nElapsed time: ", elapsed_time, "\n\n")
    print('\n' +"Optimizaton complete" + '\n')



    

    