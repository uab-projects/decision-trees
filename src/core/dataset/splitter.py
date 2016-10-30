# libraries
import random

"""
Given a dataset, generates a training set and a validation set according to
the splitting method specified
"""
class DatasetSplitter(object):
    """
    @attr   _dataset    original dataset
    """
    __slots__ = ["_dataset"]

    """
    Initializes a new dataset splitter with the dataset to split specified as
    a parameter
    """
    def __init__(self, dataset):
        self._dataset = dataset

    """
    Generates a training set and a validation set using the holdout method, this
    means, specifying a percent of the dataset, we will generate a random
    training set that contains that percentage of the dataset and the rest will
    be the validation set

    @param  percent     percent of random samples to pick from the dataset to
                        set as the training set
    @return trainingSet and validationSet objects generated
    """
    def holdout(self, percent):
        # shuffle dataset data
        
