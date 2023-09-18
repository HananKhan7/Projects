# Importing libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow import keras
from keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping


class ML_models():
    """
    Class of ML models used to predict churn.
    """
    def __init__(self):
        """
        Constructor method for the class of ML models, which stores the data.
        """
        self.X_train,self.X_train_split, self.X_validation, self.X_test,self.y_train,\
              self.y_train_split, self.y_train_original, self.y_validation, self.y_test, self.y_test_original = self.data_preprocessing()

    def data_preprocessing(self):
        # Load processed data with churn variable
        data = shuffle(pd.read_csv('dataset/data_churn.csv').drop("Unnamed: 0", axis=1))
        x_variables = ['dateCreated', 'page_turn_count', 'view_duration','InstallDate']
        X = data[x_variables]
        y = data['churn'].values

        # One-hot encoder for churn variable (3 classes --> low, mid, and high risk)
        ohe = OneHotEncoder(sparse=False)
        y_encoded = ohe.fit_transform(y.reshape(-1, 1))

        # data split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=50)
        # Standardizing data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)     
        X_test = scaler.transform(X_test) 
        # Further splitting train data into training and validation data (For NN early stopping criteria)
        X_validation, X_train_split = np.split(X_train,[int(0.1*len(X_train))])                # 10 % training data for validation, Rest is used for traininig.
        y_validation, y_train_split = np.split(y_train,[int(0.1*len(y_train))])   
        # Reversing one hot encoder transformation (For evaluation and SVM model)
        y_train_original = ohe.inverse_transform(y_train).ravel()
        y_test_original = ohe.inverse_transform(y_test).ravel()
        return X_train, X_train_split, X_validation, X_test, y_train, y_train_split, y_train_original, y_validation, y_test, y_test_original 

    def neural_network(self):
        """ 
        This method builds, trains, and evaluates a NN from keras
        """
        # Initializing the Neural Network
        nn = keras.models.Sequential()
        # Adding input layer
        nn.add(keras.layers.Input(shape=(self.X_train.shape[1],)))               # Adding number of neurons based on selected relevant variables)
        # Adding Hidden layers
        nn.add(keras.layers.Dense(units =50 , activation = 'PReLU', kernel_regularizer = l1(0.01)))
        nn.add(keras.layers.Dense(units =50 , activation = 'PReLU', kernel_regularizer = l2(0.01)))
        nn.add(keras.layers.Dense(units = 3 , activation = 'softmax'))      # 3 categories of risk based on churn prediction (High, mid, and low risk)
        nn.compile(optimizer='adam',loss='categorical_crossentropy', metrics='accuracy') 
        # Training NN
        # Adding early stopping
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
        # Setting custom weights, to increase training prirority for page turn counts and view duration based on data analysis.
        nn.fit(self.X_train_split, self.y_train_split, batch_size=32, epochs = 15 ,verbose = 1, validation_data = [self.X_validation, self.y_validation], callbacks = [early_stopping])
        # Evaluate the model
        _, nn_accuracy = nn.evaluate(self.X_test, self.y_test)
        nn_pred = nn.predict(self.X_test)
        nn_pred_classes = np.argmax(nn_pred, axis=1)
        return nn_accuracy, nn_pred_classes
    
    def random_forest(self):
        """
        This method builds, trains, and evaluates a Random forest classifier from sklearn.
        """
        rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=50,n_jobs=10,verbose=1)
        # Training random forest
        rf.fit(self.X_train, self.y_train)
        # predicting results
        rf_pred = rf.predict(self.X_test)
        rf_pred_classes = np.argmax(rf_pred, axis=1)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        return rf_accuracy, rf_pred_classes
    
    def svm(self):
        """
        This method builds, trains, and evaluates a Support vector classifier from sklearn.
        """
        svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr', random_state=50)
        # Training svm
        svm.fit(self.X_train, self.y_train_original)
        # predicting results
        svm_pred = svm.predict(self.X_test)
        svm_accuracy = accuracy_score(self.y_test_original, svm_pred)
        return svm_accuracy, svm_pred
    
    def model_eval(self, model):
        """
        This method evalutes the ML models used and prints their accuracy score, classification report, and
        confusion matrix into a text file --> "Evaluation_scores.txt"
        """
        accuracy, pred_classes = model()
        report = classification_report(pred_classes, self.y_test_original, target_names = ["low Risk", "Mid Risk", "High Risk"])
        cm = confusion_matrix (pred_classes, self.y_test_original)
        if 'neural_network' in str(model):
            model_name = 'neural_network' 
        if 'random_forest' in str(model):
            model_name = 'random_forest' 
        if 'svm' in str(model):
            model_name = 'svm' 
        with open('evaluation_results/{}_scores.txt'.format(model_name), 'w') as f:
            #  Accuracy score
            f.write("ACCURACY SCORE\n\n")
            f.write(f"model accuracy: {accuracy:.4f}\n")
            # Classification report
            f.write("\n\nCLASSIFICATION REPORT \n\n")
            f.write("\n\n {}".format(report))   
            # Confusion matrix
            f.write("\n\nCONFUSION MATRIX\n\n")
            f.write("\n\n {}".format(cm)) 
            f.close()
            
    def ML_model_selection(self,model_selection:str):
        """
        parameter:
        model_selection: Can select out of three ML model options. Option to choose (1,2 or all 3 ML models)
        'nn' --> neural network
        'rf' --> random forest classifier
        'svm' --> support vector classifier
        e.g: model_selection: 'nn', or 'nn rf', or 'rf svm', or 'nn rf svm' etc.
        """
        # Generating evaluation 
        if os.path.isdir('evaluation_results') == False:
            os.mkdir('evaluation_results')
        
        if 'nn' in model_selection:
            self.model_eval(self.neural_network)

        if 'rf' in model_selection:
            self.model_eval(self.random_forest)
            
        if 'svm' in model_selection:
            self.model_eval(self.svm)

        if 'nn' and 'rf' and 'svm' not in model_selection:
            print("Please enter correct ML model name out of these options --> 'nn', 'rf', 'svm'")
 
if __name__ == "__main__":
    ml = ML_models()
    ml.ML_model_selection('nn rf svm')