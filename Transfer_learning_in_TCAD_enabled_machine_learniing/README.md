# Transfer learning in TCAD enabled machine learning

## Introduction
Transfer learning stands for utilizing feature representations from a pre-trained model so that we do not have to train a new model from scratch, and it comes in handy when we do not have a huge data set. 

In this project, Transfer learning was used to aid in the training process (reduce the amount of training data required), by transferring device physics from a highly trained ML model/ Digital twin of one semiconductor device to an untrained ML model of another yet similar semiconductor device.
## Implementation
### Base model
First step is to select a highly trained and optimized ML model of a similar device. Here similar means, a ML model having common input and output parameters. As shown below, the model on the left is trained and optimized, it serves as a base model to transfer device physics information to second/new model for a similar yet different device.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Transfer_learning_in_TCAD_enabled_machine_learniing/plots/transfer_learning_general.png)
### Weight vectors
Transfer learning is mainly taking place in the first two layers of ML model, input and hidden layer. Weight vector manipulation is used to conduct transfer learning.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Transfer_learning_in_TCAD_enabled_machine_learniing/plots/hidden_layer_mathematical_explanation.png)
### Setting up features/inputs
Based on the use case, weight vectors of features that are common between base and second model are transferred. Un wanted feature's weight vectors, unique to base model,  are removed and features unique to second/new model are added, initially as random starting weights.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Transfer_learning_in_TCAD_enabled_machine_learniing/plots/Adding_removing_feature.png)
### Training process
#### Freezing weights
Initially in the training process, transferred weight vectors from base model are frozen, while new feature weight vectors are left untouched. This allows the ML model to specifically train and learn weight vectors of new features w.r.t the rest of transferred feature weight vectors, without altering transferred feature weight vectors. 

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Transfer_learning_in_TCAD_enabled_machine_learniing/plots/freezing_weights.png)
#### Un-freezing weights
After the first training process, transferred feature weight vectors are unfrozen, and with a low learning rate, training process is repeated. This allows fine tuning and re calibration.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Transfer_learning_in_TCAD_enabled_machine_learniing/plots/unfreezing_weights.png)
### Results
With the help of transfer learning, ML models were able to perform with the same accuracy as before but with 50 % of the training data required as compared to the standard training procedure. This was quite beneficial, as training data required to create a digital twin of a semiconductor device is computationally expensive.

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Transfer_learning_in_TCAD_enabled_machine_learniing/plots/TL_results.png)
