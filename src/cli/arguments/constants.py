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

"""
Filters available
"""
FILTERS = ["none","remove-unknown-rows"]
FILTERS_DEFAULT = FILTERS[0]

# Tree caching
"""
Caches the generated tree into the following file
"""
TREE_CACHE_FILE = ".cached-trees.json"

# Random forest
"""
Random (corre) forest algorithm enabled by default
"""
RANDOM_FOREST_DEFAULT = False

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
ALGORITHM_DEFAULT = ALGORITHMS[1]

"""
Sets the algorithm to use to split a dataset into a dataset and training set if no training set is available
"""
SPLITTERS = ["holdout","cross-validation","leave1out","bootstrapping"]
SPLITTER_DEFAULT = SPLITTERS[0]

"""
Default percentage of items that will be treated as training set if no
validation set is specified, when holdout method is specified
"""
HOLDOUT_PERCENT_DEFAULT = 0.75

"""
Number of groups to create to use the cross-validation method to split the dataset into training and validation sets
"""
CROSSVALID_K = 4
CROSSVALID_K_MIN = 2

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
