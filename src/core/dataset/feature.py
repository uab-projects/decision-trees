# app modules
from core.helpers.math import H

# libraries
import logging

# constants
LOGGER = logging.getLogger(__name__)


"""
Defines a feature for a machine learning dataset, including its name, the values it can take, if it's continuous or not...
"""
class Feature(object):
	"""
	@attr   _index			number of the feature (column in the dataset)
	@attr   _name			name of the feature if exists
	@attr   _values			list of values the feature can take
	@attr   _valuesType 	type of the values of the feature can take
							it should be a function to convert it to that type
	@attr 	_valuesNames 	the names for each feature values
	@attr   _isContinuous	determines if the feature can take continuous values
	"""
	__slots__ = ["_index","_name","_values","_valuesType","_isContinuous"]

	"""
	Initialiazes a new feature given it's index, the values it can take, a boolean indicating if it's continuous or not, the type for that values if necessary, and the name if provided

	@param 	index 		numerical index of the feature inside the dataset
	@param 	values 		values the feature can take
	@param 	continuous 	true if continuous feature (false by default)
	@param 	valuesType	type of the values the feature can take
	@param 	name 	name of the feature if exists
	"""
	def __init__(self, index, values, continuous = False, valuesType = str, name = None):
		self._index = index
		self._values = values
		self._continuous = continuous
		self._valuesType = valuesType
		self._guessValuesNames()
		self._name = name
		if name == None:
			self._name = "Feature <%02d>"%self._index

	"""
	Will set the values names as their default values if no values names are specified yet
	"""
	def _guessValuesNames(self):
		self._valuesNames = ["Value <%s>"%str(self._valuesType(value)) for value in self._values]

	def getIndex(self):
		return self._index

	def getValues(self):
		return self._values

	def setValues(self, values):
		self._values = values
		if len(self._values) != len(self._valuesNames):
			self._guessValuesNames()

	def isContinuous(self):
		return self._isContinuous

	def getValuesType(self):
		return self._valuesType

	def getValuesNames(self):
		return self._valuesNames

	def setValuesNames(self, valuesNames):
		self._valuesNames = valuesNames

	def getName(self):
		return self._name

	def setName(self, name):
		self._name = name

class DataFeature(Feature):
	"""
	@attr 	_data 		data of the feature
	"""
	__slots__ = ["_data"]

	"""
	Initialiazes a new feature given it's index, the values it can take, a boolean indicating if it's continuous or not, the type for that values if necessary, and the name if provided

	@param 	index 		numerical index of the feature inside the dataset
	@param 	data 		data of the feature
	@param 	values 		values the feature can take
	@param 	continuous 	true if continuous feature (false by default)
	@param 	valuesType	type of the values the feature can take
	@param 	name 	name of the feature if exists
	"""
	def __init__(self, index, data, values, continuous = False, valuesType = str, name = None):
		super().__init__(index, values, continuous, valuesType, name)
		self._data = data

	def setData(self, data):
		self._data = data

	def getData(self):
		return self._data


class ClassifierFeature(DataFeature):
	"""
	@attr 	entropy 	entropy of the feature
	@attr   threshold	if it's continuous, the threshold to separate the continuous values
	"""
	__slots__ = ["_entropy", "_threshold"]

	def __init__(self, index, data, values, continuous = False, valuesType = str, name = None):
		super().__init__(index, data, values, continuous, valuesType, name)
		self.calculateEntropy()

	def getEntropy(self):
		return self._entropy

	"""
	Calculates the entropy of the feature for the current samples in the training set

	@param 	trainingSet 	samples we're mananing now
	@param 	targetData 		data of the target to count classes
	"""
	def calculateEntropy(self, trainingSet, targetData):
		acc = 0.0
		for value in self._values:
			filteredTrainingSet = trainingSet * np.where(self._data == value,True,False)
			targetCount = np.bincount(targetData[filteredTrainingSet],minlength=len(self._values))
			acc += np.sum(filteredTrainingSet)/np.sum(trainingSet) * H(targetCount)
		return acc

	def getThreshold(self):
		return self._threshold

	def setThreshold(self, threshold):
		self._threshold = threshold
