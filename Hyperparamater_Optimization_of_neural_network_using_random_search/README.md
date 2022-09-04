# Hyperparameter Optimization of a neural network model using random search

## Introduction
Hyper parameter optimization is the process of for finding the best hyperparameters for a NN model with respect to a certain dataset. There is no direct way of finding the optimal hyperparameters with respect to a given dataset. Finding the best hyperparameters can be tedious if done by hit and trial method. This random search implementation finds the optimal hyperparameters for with respect to a given dataset by comparing the mean absolute error scores from multiple NN configurations.
## Flow diagram
Figure below shows the process for finding the optimal hyperparameters.
- Based on the defined search space, model creates a configuraion.
- Since there are multiple configurations that the model will check, It is ideal to create to check for repetitions, in order to only generate unique configurations.
- Model is trained on the same HP configuration multiple times in order to check model variance and consistency of the configuration itself.
- MAE (Mean absolute error) scores of all the generated configurations are appended in a single excel sheet, which is also used to check for repetitions. 

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Hyperparamater_Optimization_of_neural_network_using_random_search/plots/Flow_diagram.png)

## Output
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Hyperparamater_Optimization_of_neural_network_using_random_search/plots/Excel%20output.png)

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Hyperparamater_Optimization_of_neural_network_using_random_search/plots/hl_score_plot.png)

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Hyperparamater_Optimization_of_neural_network_using_random_search/plots/box%20plot.png)

