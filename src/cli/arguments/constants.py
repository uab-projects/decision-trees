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
ALGORITHMS = ["id3","dummy"]
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
