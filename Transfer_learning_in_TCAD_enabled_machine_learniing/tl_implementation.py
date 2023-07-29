# Importing Libraries
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
import os,re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # To suppress tensorflow warnings
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm
import random
import os.path
import pickle
from scipy.spatial.distance import pdist

def data_import_and_preprocess(filepath:str):
    """
    parameter: filepath which is the directory of the datafile.

    This method imports the data file, removes the first column which is index column
    and returns the shuffles the data
    """
    data = pd.read_csv(filepath)
    data = shuffle(data.dropna())
    return data

def generate_eval_data(flavor: str,tcad_data, axial_data, features, no_eval_data: int):
    for i in range(len(axial_data)):
        if re.findall('target', axial_data['ctrlstr'].iloc[i]):
            target_data = axial_data[features].iloc[i].to_list()
    inpars = pd.DataFrame()
    inpars['feature'] = features
    inpars['target'] = target_data

    df0=tcad_data
    mycols=['ctrlstr']+features
    #normalize ranges
    df1=df0[mycols].copy()
    for mycol in features:
        df1[mycol]=(df0[mycol]-df0[mycol].min())/(df0[mycol].max()-df0[mycol].min())
    #Maximize minimum distance between items to make sample as uniform as possible
    iterations=300
    mindist=0
    minlist=[]
    maxpos=[]
    for ii in range(iterations):
        sampledf=df1.sample(n=no_eval_data)
        distances=pdist(sampledf[features].to_numpy())
        minlist.append(distances.min())
        if mindist < minlist[-1]:
            maxpos.append(ii)
            mindist=minlist[-1]
        seldf=sampledf.copy()
    minlist=np.array(minlist)
    #save selected df to disk
    seldf.sort_index(inplace=True)
    expdf=df0.iloc[seldf.index].copy()
    expdf = expdf.dropna()
    expdf.to_csv("{}_model/evaluation_data.csv".format(flavor),index=False)
    return expdf

""" Threshold evaluation"""
def best_models(models:pd.DataFrame, base_flavor:str, second_flavor:str, thresholds:dict):
    """
    Parameters:
        models: This is a dataframe containing top performing models obtained from HPO (Hyper parameter Optimization)
        This method checks the last model added into the dataframe: models based on the threshold criteria. 
            For each label --> (absolute error/ absolute threshold) 
    """
    # Finding index of models passing the threshold evaluation
    passed_labels, failed_labels = {},{}
    passed = True
    for label in thresholds.keys():
        score = (models.iloc[len(models)-2]['{}_abs99%'.format(label)] / thresholds[label][1])
        if (score < 1):
            passed_labels['{}_abs99%'.format(label)] = score
            passed
        else:
            failed_labels['{}_abs99%'.format(label)] = score
            passed = False
    if passed:
        #print("\n....................... Model Passed threshold evaluation .......................")
        with open('TL_trained_models/{}_to_{}/evaluation_result.txt'.format(base_flavor, second_flavor), 'w') as f:
            f.write('Evaluation Result: Passed\n')
            for label_name in passed_labels.keys():
                f.write('{} score: {}\n'.format(label_name, passed_labels[label_name]))
            f.close()
    else:
        #print("\n....................... Model Failed threshold evaluation .......................")
        print(failed_labels)
        with open('TL_trained_models/{}_to_{}/evaluation_result.txt'.format(base_flavor, second_flavor), 'w') as f:
            f.write('Evaluation Result: Failed\n')
            f.write('Failed outputs:\n')
            for label_name in failed_labels.keys():
                f.write('{} score: {}\n'.format(label_name, failed_labels[label_name]))
            f.close()
    return passed


class Transfer_learning:
    def __init__(self, base_flavor:str, second_flavor:str, no_training_data: int, no_eval_data:int, training_repetition: int, features:dict, labels:list, thresholds:dict, step_size:int) -> None:
        self.base_flavor = base_flavor
        self.second_flavor = second_flavor
        self.no_training_data = no_training_data
        self.no_eval_data = no_eval_data
        self.training_repetition = training_repetition
        self.features = features
        self.labels = labels
        self.thresholds = thresholds
        self.step_size = step_size
        self.data_generator()
        self.seed, self.X_train, self.X_test, self.Y_train, self.Y_test, self.sc, self.sc_output = self.data_import()                
        self.base_model, self.layer_dimensions = self.import_model()
        self.second_model = self.clonning_model()
        self.unique_features = self.unique_features_idf()

    def import_model(self):
        """ Importing base model (NN structure) """
        itemList =  os.listdir('{}_model'.format(self.base_flavor))
        identifier_model = re.compile('.*json')
        identifier_weights = re.compile('.*h5')
        model_name = list(filter(identifier_model.match,itemList))[0]
        weight_name = list(filter(identifier_weights.match,itemList))[0]
        #Reading the model from JSON file
        with open('{}_model/{}'.format(self.base_flavor,model_name)) as json_file:
            json_savedModel= json_file.read()
        #load the model architecture 
        base_model = tf.keras.models.model_from_json(json_savedModel)
        #base_model.summary()
        # Load weights
        base_model.load_weights('{}_model/{}'.format(self.base_flavor,weight_name))
        """ Extracting hidden layer dimensions """
        layer_dimensions = [] 
        for layer in base_model.layers[:-1]:        # Ignoring output layer from base model
            layer_dimensions.append(layer.get_output_at(0).get_shape()[1])
        return base_model, layer_dimensions

    def data_generator(self):
        """ Splitting TCAD data into training only and train_test data"""
        TCAD_data = data_import_and_preprocess('{}_model/p{}.csv.gz'.format(self.second_flavor, self.second_flavor))
        train_test_index = []
        train_index = []
        for i in range(len(TCAD_data)):
            if re.findall('ae_lhsb_', TCAD_data['ctrlstr'].iloc[i]):
                train_test_index.append(i)
            else:
                train_index.append(i)
        # Splitting TCAD data
        train_test_data = TCAD_data.iloc[train_test_index].drop_duplicates()
        training_data = TCAD_data.iloc[train_index].drop_duplicates()
        # Exporting to .csv files
        train_test_path = '{}_model/train_test.csv'.format(self.second_flavor)
        training_path = '{}_model/training_data.csv'.format(self.second_flavor)
        training_data.to_csv(training_path, index = False)
        train_test_data.to_csv(train_test_path, index = False)

    def data_import(self):
        """ Importing TCAD data (For training new model)"""
        # Setting and saving random seed
        seed = random.randint(1,2000)
        tf.random.set_seed(seed)
        # Removing evaluation data from TCAD_data ------> Training data, which is further narrowed down to the number of training samples.
        # Removing duplicates by 'ctrlstr' column
        TCAD_data = pd.read_csv('{}_model/train_test.csv'.format(self.second_flavor))
        axial_data = pd.read_csv('{}_model/training_data.csv'.format(self.second_flavor))
        evaluation_data = generate_eval_data(self.second_flavor, TCAD_data, axial_data, self.features[self.second_flavor], self.no_eval_data)
        training_data = TCAD_data.copy()
        duplicate_index = TCAD_data['ctrlstr'].isin(evaluation_data['ctrlstr'])
        training_data.drop(training_data[duplicate_index].index, inplace = True)
        # Drop simulations with 0 values
        axial_data = axial_data[self.features[self.second_flavor]+self.labels].dropna()
        training_data = training_data[self.features[self.second_flavor]+self.labels].dropna()
        evaluation_data = evaluation_data[self.features[self.second_flavor]+self.labels].dropna()
        X_train = pd.concat([axial_data[self.features[self.second_flavor]],training_data.iloc[0:self.no_training_data][self.features[self.second_flavor]]], ignore_index=True, sort=False)
        Y_train = pd.concat([axial_data[self.labels],training_data.iloc[0:self.no_training_data][self.labels]], ignore_index=True, sort=False)
        X_test = evaluation_data[self.features[self.second_flavor]]
        Y_test = evaluation_data[self.labels]
        # Scaling the input data using Standard Scaler
        sc = StandardScaler()
        sc_output = StandardScaler()
        X_train = sc.fit_transform(X_train.values)
        X_test = sc.transform(X_test.values)
        Y_train = sc_output.fit_transform(Y_train.values)
        Y_test = sc_output.transform(Y_test.values)
        # We dont apply fit method to the test data so that the model does not compute mean standard deviation
        # of the test dataset during training
        return seed, X_train, X_test, Y_train, Y_test, sc, sc_output

    def clonning_model(self):
        """ clonning base model """
        clonned_model = tf.keras.Sequential()
        # Adding input layer
        clonned_model.add(tf.keras.layers.Input(shape=(len(self.features[self.second_flavor]),)))
        # Adding hidden layers
        for dimensions in self.layer_dimensions:
            clonned_model.add(tf.keras.layers.Dense(units =dimensions , activation = 'PReLU')) 
        # Adding output layer
        clonned_model.add(tf.keras.layers.Dense(units=len(self.labels), activation = 'linear'))
        return clonned_model

    def unique_features_idf(self):
        """ Transferring weight vectors of common features from base model """
        unique_features = {}            # Creating a dictionary to store unique features in both base and second model, along with their indexes.
        unique_features[self.base_flavor] = [math.nan, math.nan]
        unique_features[self.second_flavor] = [math.nan, math.nan]
        # For base model
        for i in range(len(self.features[self.base_flavor])):
            if self.features[self.base_flavor][i] in self.features[self.second_flavor]:
                pass
            else:
                unique_features[self.base_flavor] = [self.features[self.base_flavor][i], i]
        # For second model
        for i in range(len(self.features[self.second_flavor])):
            if self.features[self.second_flavor][i] in self.features[self.base_flavor]:
                pass
            else:
                unique_features[self.second_flavor] = [self.features[self.second_flavor][i], i]
        return unique_features

    def weight_transfer(self,function_call:str):
        # For clonning weights of base model
        base_weights = np.zeros([len(self.features[self.base_flavor]),self.layer_dimensions[0]])
        second_weights = np.zeros([len(self.features[self.second_flavor]),self.layer_dimensions[0]])
        # Storing weights
        weights_base = self.base_model.layers[0].weights[0]
        weights_second = self.second_model.layers[0].weights[0]
        # Clonning
        for i in range(len(self.features[self.base_flavor])):
            np.copyto(base_weights[i], weights_base[i], casting='same_kind', where=True)
        for i in range(len(self.features[self.second_flavor])):
            np.copyto(second_weights[i], weights_second[i], casting='same_kind', where=True)
        # converting to list
        weights_list_base = base_weights.tolist()
        weights_list_second = second_weights.tolist()
        second_weights = []
        for i in range(len(self.features[self.second_flavor])):
            if self.features[self.second_flavor][i] in self.features[self.base_flavor]:
                # Getting index of this common feature in base model to transfer its weights, temporarily converting to list since dictionary has no order so it does not store index.
                second_weights.append(weights_list_base[list(self.features[self.base_flavor]).index(self.features[self.second_flavor][i])]) 
            # For features unique to second flavour, training is required, so for now, we keep the standard initial weights of new model. (or learned weights if its a second call of this function)
            if self.features[self.second_flavor][i] not in self.features[self.base_flavor]:
                second_weights.append(weights_list_second[i])            
        # Converting weights back to np.array and then to tensor.
        second_weights_np = []
        for i in range(len(second_weights)):
            second_weights_np.append(np.array(second_weights[i]))
        second_weights_tf = tf.convert_to_tensor(second_weights_np)
        # Clonning bias and activation function values
        updated_weights = []
        updated_weights.append(second_weights_tf)
        # If its the function call (no training yet), then we transfer bias values from base model. If its second call (After training),
        # then we keep learned bias values of new model. Activation function values are learned from scratch.
        if function_call == 'first':
            updated_weights.append(self.base_model.layers[0].weights[1])
            updated_weights.append(self.second_model.layers[0].weights[2])          # Keep activation function values to default intial values, for training.
        else:
            updated_weights.append(self.second_model.layers[0].weights[1])
            updated_weights.append(self.second_model.layers[0].weights[2])  
        return updated_weights

    def learning_weights(self):
        """CASE 1: If there are unique features in second flavor, then we transfer weight vectors of common features and then 'partial train' new model to get weights for unique features. """
        loop_counter = 0
        if self.unique_features[self.second_flavor] != [math.nan, math.nan]:
            if loop_counter == 0:
                #print("\n....................... Unique features found in second flavor, Initiating training process .......................")
                updated_weights = self.weight_transfer('first')
                # Transferring weights to new model.
                self.second_model.layers[0].set_weights(updated_weights)
                # Training model to learn weights of feature unique to second flavor.
                self.second_model.compile(optimizer='adam', loss = 'mse', metrics = ['mae'])
                self.second_model.fit(self.X_train, self.Y_train, batch_size=32, epochs = 500, verbose = 0)
                #print("\n....................... Initial training complete, transferring weight vectors .......................")
                # For common feautures, we re transfer the weight vectors from base model, while for features unique to second flavor, we keep the learned weights.
                updated_weights = self.weight_transfer('second')
                # Transferring weights to new model.
                self.second_model.layers[0].set_weights(updated_weights)
                loop_counter+=1
        """ CASE 2:  If there are no unique features in second flavor, then we only need to copy weight vectors of common features. """
        if self.unique_features[self.second_flavor] == [math.nan, math.nan]:
            if loop_counter == 0:
                #print("\n....................... No uniqe features found in second flavor, training not required, transferring weights from base model .......................")
                updated_weights = self.weight_transfer('first')
                # Transferring weights to new model.
                self.second_model.layers[0].set_weights(updated_weights)
                loop_counter+=1

        # In case we have multiple hidden layers, we just copy weight vectors along with biasvalues from base model. (prev version of transfer learning)
        if len(self.layer_dimensions) > 1:
            for i in range(len(self.layer_dimensions)-1):
                self.second_model.layers[i+1].set_weights(self.base_model.layers[i+1].weights)

    def new_model_gen(self):
        """ Transferring all learned information to new_model """
        new_model = tf.keras.Sequential()
        # Adding input layer
        new_model.add(tf.keras.layers.Input(shape=(len(self.features[self.second_flavor]),)))
        # Adding hidden layers
        for dimensions in self.layer_dimensions:
            new_model.add(tf.keras.layers.Dense(units =dimensions , activation = 'PReLU')) 
        # Adding output layer
        new_model.add(tf.keras.layers.Dense(units=len(self.labels), activation = 'linear'))
        # For common feautures, we re transfer the weight vectors from base model, while for features unique to second flavor, we keep the learned weights.
        updated_weights = self.weight_transfer('second')
        # Transferring weights to new model.
        new_model.layers[0].set_weights(updated_weights)
        return new_model
    
    def fine_tuning(self, new_model):
        """ Training new model """
        #print("\n....................... Initiating Final training/ Fine-tuning .......................")
        # Training model number of times based on training repetitions variable.
        for _ in range(self.training_repetition+1):
            #Setting learning rate to very low value
            adam_opt = tf.keras.optimizers.Adam(learning_rate= 0.001)
            new_model.compile(optimizer=adam_opt, loss = 'mse', metrics = ['mae'])
            new_model.fit(self.X_train, self.Y_train, batch_size=32, epochs = 500, verbose = 0) 
        return new_model
    
    """ Model Evaluation"""
    
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
        percentile_99th = []
        for i in range(true_values.shape[1]):   # Number of labels
            percentage_error_per_label = []
            absolute_error_per_label = []
            for row_number in range(true_values.shape[0]):
                # if true_values[row_number][i] != 0 and pred_values_transform[row_number][i] != 0:
                percentage_error_per_label.append(abs(true_values[row_number][i] - pred_values_transform[row_number][i])/((true_values[row_number][i] + pred_values_transform[row_number][i])/2)*100)
                absolute_error_per_label.append(abs(true_values[row_number][i] - pred_values_transform[row_number][i]))
            percentage_error_per_label.sort()
            absolute_error_per_label.sort()
            percentile_99th.append(absolute_error_per_label[round(len(absolute_error_per_label) * 0.99)])         # Trying with 99th percentile
        return [percentile_99th]

    def evaluate_model(self,new_model):
        #print("\n....................... Evaluating Model .......................")
        """
        This method prints the loss (mean absolute error) computed from the trained model.
        parameter:
        nn: trained neural network
        """
        predicted_values = new_model.predict(self.X_test)
        score = self.error_per_label(predicted_values)
        loss = new_model.evaluate(self.X_test ,self.Y_test, verbose=0)
        score.append(loss[1])         # To return 99th percentile abs error per label and overall MAE_score
        return score  

    """ Storing scores """

    def output_label_gen(self):
        """
        parameter: labels/ electrical parameter names

        This method generates a set of labels for each electrical parameter. One representing percentage scores
        and the second representing absolute differences between true and predicted label values
        """
        labels_for_out = []
        for label in self.labels:
            labels_for_out.append(label+"_abs99%")       # To represent 99th percentile absolute error
            # labels_for_out.append(label+"_abs")     # To represent absolute 
        return labels_for_out

    def output_score_format(self, scores):
        """
        parameter: Percentage and absoulte difference scores for each label

        This method is used to set the format/order of the scores to a more presentable one.
        e.g : Output ---> label_1_%, label_1_abs, label_2_%, label_2_abs,....
        """
        percentile_score = scores[0]
        # abs_score = scores[1]
        output_scores = []
        for i in range(len(percentile_score)):
            output_scores.append(percentile_score[i])
            # output_scores.append(abs_score[i])
        return output_scores
        
    def output_to_excel(self, model_score):
        """
        parameter: 
            A list of lists containing NN strucutre data (Number of hidden layers, Number of neurons, MAE values and runtime information)
            output path for the excel sheet containing model score and info.
        
        This method creates a dataframe based on the output list from the class and exports it into .csv file (Excel sheet) 
        """
        output_labels = self.output_label_gen()
        output_scores = self.output_score_format(model_score)
        df = pd.DataFrame()
        df["Number of hidden layers"] = [len(self.layer_dimensions)]
        df["Number of neurons"] = [self.layer_dimensions]
        df["Amount of training data"] = [self.no_training_data]
        df["Norm_MAE_avg"] = model_score[1]
        for i in range(len(output_labels)):        # For total number of scores
            df[output_labels[i]] = output_scores[i]
        # Check if directory exists
        if os.path.isdir('TL_trained_models') == False:
            os.mkdir('TL_trained_models')
        if os.path.isdir("TL_trained_models/{}_to_{}".format(self.base_flavor,self.second_flavor)) == False:
            os.mkdir("TL_trained_models/{}_to_{}".format(self.base_flavor,self.second_flavor))              
        # Saving the dataframe to an excel file
        output_path = "TL_trained_models/{}_to_{}/{}_model_score.csv".format(self.base_flavor, self.second_flavor, self.second_flavor)

        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))		# This will append the datafile and put headers only for the first time.
        return df

    def saving_model(self, new_model):
        """ Saving trained model """
        #print("\n....................... Saving model .......................")
        model_json = new_model.to_json() 
        # Saving seed
        with open('TL_trained_models/{}_to_{}/seed.txt'.format(self.base_flavor,self.second_flavor), 'w') as f:
            f.write('Seed : {}'.format(self.seed))    
        # Saving model to JSON
        with open("TL_trained_models/{}_to_{}/TL_{}.json".format(self.base_flavor, self.second_flavor, self.second_flavor), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        new_model.save_weights("TL_trained_models/{}_to_{}/TL_{}.h5".format(self.base_flavor, self.second_flavor, self.second_flavor),"w")
        # Saving scalers
        pickle.dump(self.sc, open('TL_trained_models/{}_to_{}/scaler_features.pkl'.format(self.base_flavor, self.second_flavor), 'wb'))
        pickle.dump(self.sc_output, open('TL_trained_models/{}_to_{}/scaler_labels.pkl'.format(self.base_flavor, self.second_flavor), 'wb'))  

    def TL_main(self):
        """
        This method implements transfer learning till the least amount of training data required for passing the evaluation criteria
        is found.
        """
        print("\n....................... Initiating TL optimizer .......................")
        loop_counter = 1
        evaluation_status = True         
        while evaluation_status:
            self.learning_weights()
            new_model = self.new_model_gen()
            trained_model = self.fine_tuning(new_model)
            model_score = self.evaluate_model(trained_model)
            output_df = self.output_to_excel(model_score)
            evaluation_status = best_models(output_df, self.base_flavor, self.second_flavor, self.thresholds)
            print("\n....................... Iteration number: {} with {} training data .......................".format(loop_counter, self.no_training_data))
            if evaluation_status:
                print("\n....................... Model passed evaluation criteria, reducing training data .......................".format(evaluation_status))
            if evaluation_status == False:
                print("\n....................... Evaluation failed, saving previous model .......................")
                print("\n....................... Model found with least training data {} after {} iterations".format(self.no_training_data, loop_counter))
                print("\n....................... Saving model .......................")
                self.saving_model(previous_model)
                break
            if self.no_training_data<=10:
                print("\n....................... Evaluation passed with no training data, saving model .......................")
                print("\n....................... Model found with least training data {} after {} iterations".format(self.no_training_data+10, loop_counter))
                print("\n....................... Saving model .......................")
                self.saving_model(previous_model)
                break
            previous_model = trained_model
            self.no_training_data = self.no_training_data-self.step_size            # Reducing training data size
            loop_counter +=1


