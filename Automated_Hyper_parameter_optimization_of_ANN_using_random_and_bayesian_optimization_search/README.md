# Automated hyper parameter Optimization of an artificial neural network using random and bayesian optimization search

## Introduction
Hyper parameter optimization is the process of for finding the best hyperparameters for a NN model with respect to a certain dataset. Finding the best hyperparameters can be tedious if done by hit and trial method. Automated hyper parameter optimization is conducted in this project, using a combination of random and bayesian optimization search.
## Flow diagram
ADD PLOTS WITH DESCRIPTION AS IN TL PROJECT.
 
Figure below shows the process for finding the optimal hyperparameters.
- The model starts off with random search.
- Based on the defined search space, model creates a configuraion.
- Since there are multiple configurations that the model will check, It is ideal to create to check for repetitions, in order to only generate unique configurations.
- Model is trained on the same HP configuration multiple times in order to check model variance and consistency of the configuration itself.
- MAE (Mean absolute error) scores of all the generated configurations are appended in a single excel sheet, which is also used to check for repetitions.
- This random search is parallelized, to allow the model to check multiple configurations of hyper parameters, at the same time, based on available resources.
- After completion of random search, model configurations with best scores are forwarded to bayesian optimization search, as inputs.
- With the help of bayesian optimization, a narrow search space is explored around the best found configurations from random search.
- All generated models are saved in a seperate folder along with their weights,scalers and random seeds, automatically.
- Final output contains a group of high performing model configurations.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Automated_Hyper_parameter_optimization_of_ANN_using_random_and_bayesian_optimization_search/plots/Flow_diagram.png)

## Output
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Automated_Hyper_parameter_optimization_of_ANN_using_random_and_bayesian_optimization_search/plots/Excel%20output.png)

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Automated_Hyper_parameter_optimization_of_ANN_using_random_and_bayesian_optimization_search/plots/hl_score_plot.png)

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Automated_Hyper_parameter_optimization_of_ANN_using_random_and_bayesian_optimization_search/plots/box%20plot.png)

