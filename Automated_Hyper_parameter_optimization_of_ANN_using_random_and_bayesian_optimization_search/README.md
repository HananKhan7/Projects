# Automated hyper parameter Optimization of an artificial neural network using random and bayesian optimization search

## Introduction
Hyper parameter optimization is the process of for finding the best hyperparameters for a NN model with respect to a certain dataset. Finding the best hyperparameters can be tedious if done by hit and trial method. Automated hyper parameter optimization is conducted in this project, using a combination of random and bayesian optimization search.
## Implementation
### Wide range random search
The model starts off with random search. Initially a hyperparameter search space is defined consisiting of ranges for hyperparameters to be tuned/optimzed.
- Based on the defined search space, model selects a random group of hyperparameter values.
- Since there are multiple configurations that the model will check, a check for repetitions is created, in order to only generate unique configurations.
- Model is trained on the same HP configuration multiple times in order to check model variance and consistency of the configuration itself.
- MAE (Mean absolute error) is used as a measure of evaluation.
- This random search is parallelized, to allow the model to check multiple configurations of hyper parameters, at the same time, based on available resources.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Automated_Hyper_parameter_optimization_of_ANN_using_random_and_bayesian_optimization_search/extra/Flow_diagram.png)

MAE scores of all the generated configurations are appended in a single excel sheet, which is also used to check for repetitions.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Automated_Hyper_parameter_optimization_of_ANN_using_random_and_bayesian_optimization_search/extra/Excel%20output.png)

Box plot displays overall range of MAE score w.r.t different hidden layer configuration tested by the model, through its whiskers. The line in between represents median value. The void dots are outliers which can be ignored. Through this box plot, it can be noted that MAE scores are considerably decreasing up till 5 number of hidden layers, after that they are confined to a somewhat similar range.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Automated_Hyper_parameter_optimization_of_ANN_using_random_and_bayesian_optimization_search/extra/box%20plot.png)


### Narrow range bayesian optimization search
After completion of random search, model configurations with best scores are forwarded to bayesian optimization search, as inputs.
- With the help of bayesian optimization, a narrow search space is explored around the best found configurations from random search.
- All generated models are saved in a seperate folder along with their weights,scalers and random seeds, automatically.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Automated_Hyper_parameter_optimization_of_ANN_using_random_and_bayesian_optimization_search/extra/bayesian_search.png)

## Output

Final output contains a group of highly optimized ML models w.r.t to the training data, trained and ready to be utilized.



