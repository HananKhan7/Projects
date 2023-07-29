from tl_implementation import best_models, Transfer_learning



# Insert Semiconductor device names here:
base_flavor =                   # Device to get optimized NN model for transferring learned information.
second_flavor =                 # Device to transfer learned information to, from base model.
no_training_data = 250         # Amount of training data for new model (with transferred learned information)
step_size = 10                 # Amount of reduction in training data per iteration.
no_eval_data = 600             # Amount of uniformly selected data for evaluation
training_repetition = 0         # Amount of times the new model is retrained.
perform_evaluation = True

# Features & labels
features = {}

labels = []


# Evaluation
thresholds = {}                         # thresholds['label_name'] = [percentage threshold, absolute threshold]




#############################################################################################################################################################
""""""""""      Main (Do not modify)   """""""""""

T_L = Transfer_learning(base_flavor, second_flavor, no_training_data, no_eval_data, training_repetition, features, labels, thresholds, step_size)
T_L.TL_main()


