# Libraries


# Dataset related
"""
Dataset lists
"""
DATASETS = ["mushroom","adult"]

"""
Default dataset
"""
DATASET_DEFAULT = DATASETS[0]

# Printing control
"""
Enables or disables showing information about the dataset
"""
SHOW_DATASET_DEFAULT = False

"""
Enables or disables showing the decision tree to the screen
"""
SHOW_TREE_DEFAULT = True

# Algorithm
"""
Sets the algorithms available to use
"""
ALGORITHMS = ["id3","c4.5","dummy"]
ALGORITHM_DEFAULT = ALGORITHMS[0]

"""
Sets the variable of the dataset to use as the classifier for the decision tree
It has to bee a column from the training set
"""
TARGET_DEFAULT = 0

# Profiling
"""
Show timers
"""
TIMERS_DEFAULT = 0

"""
Default log level
"""
LOGS = ["debug","info","warning","error","critical"]
LOGS_LEVELS = [10,20,30,40,50]
LOG_DEFAULT = LOGS[1]

"""
Default percentage of items that will be treated as training set if no
validation set is specified
"""
TRAINING_PERCENT_DEFAULT = 0.8
