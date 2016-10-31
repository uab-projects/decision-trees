# app modules
from .trainingset import *
from .validationset import *

# libraries
import random
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

"""
Given a dataset, generates a training set and a validation set according to
the splitting method specified
"""
class DatasetSplitter(object):
	"""
	@attr   _dataset	original dataset
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

	@param  percent	 percent of random samples to pick from the dataset to
						set as the training set
	@return trainingSet and validationSet objects generated
	"""
	def holdout(self, percent):
		# shuffle dataset data
		numericDataset = self._dataset.getNumericDataset()
		data = numericDataset.getData()
		np.random.shuffle(data)
		# calculate slicing
		rows = self._dataset.getSampleCount()
		tr = int(rows*percent)
		# slice!
		feature_values = self._dataset.getFeaturesNumValues()
		LOGGER.debug("First sample of training set is: %s",data[0])
		trainingSet = TrainingSet(data[:tr],feature_values)
		if tr == len(data):
			LOGGER.warning("No validation set, as specified 100% as training data")
		else:
			LOGGER.debug("First element of validation set is: %s",data[tr])
		validationSet = ValidationSet(data[tr:],feature_values)
		# return
		return trainingSet, validationSet

	"""
	"""
	def crossValidation(self, k):
		pass
