import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

class Dataset(object):
	"""
	@attr 	_data   		data containing examples to train / validate
	@attr 	_cols 			rows in the dataset	(number of items)
	@attr 	_cols			cols in the dataset (number of features)
	@attr 	_features_mean 	list of attributes and classes names
	@attr 	_features_cnt	count of appearances of each feature
							format: [feature1,feature2,..]
								feature1: [val1_count,val2_count,...]
	@attr 	_features_vals	list containing the values suitable for each feature
	@attr 	_features_map 	creates a map between the found features values and numerical values
	@attr 	_isParsed 		true if already parsed
	"""
	__slots__ = ["_data","_rows","_cols","_features_vals","_features_mean",
	"_features_cnt","_features_map","_isParsed"]

	"""
	Initializes a dataset given the matrix of examples

	@param 	data 	data containing examples to train / validate
	"""
	def __init__(self,data):
		self._data = data
		self._removeUnknown()
		self._rows = len(self._data)
		self._cols = len(self._data[0])
		self._isParsed = False
		self._features_mean = None
		self._parse()


	"""
	Parses the dataset to find information about it:
		1. The values valid for each feature searching across the dataset

	"""
	def _parse(self):
		# Recognize features and its values
		self._features_vals = [sorted(list(set([row[j] for row in self._data])))
			for j in range(len(self._data[0]))]
		self._features_map = [range(len(vals)) for vals in self._features_vals]
		# Count attributes
		self._isParsed = True

	"""
	Given a list with the meaning for each feature and each feature's values, saves it in the dataset for future operations

	@param 	featureSet 	meaning of the features and its values
	"""
	def setFeaturesMeaning(self, featureSet):
		self._features_mean = featureSet

	"""
	Returns a tuple containing two objects: the training set and the validation
	set objects
	"""
	def getSets(self, percent):
		np.random.shuffle(self._data)
		tr = int(self._rows*percent)

		#LOGGER.info("training set has %d elements, validation set has %d elements from a total of %d " %(tr, self._rows-tr,self._rows))
		#LOGGER.info("first element of the training set is: ",self._data[0])
		#LOGGER.info("first element of the validation set is: ",self._data[tr])
		return (self._data[:tr], self._data[tr:])

	def __str__(self):
		txt =  "%s Specifications\n"%self.__class__.__name__
		txt += "------------------------------------------------------------\n"
		txt += "STAT:     %s\n"%("parsed" if self._isParsed else "init")
		txt += "SIZE:     %d x %d\n"%(self._rows,self._cols)
		txt += "HEAD:     %s\n"%(self._data[0])
		txt += "TAIL:     %s\n"%(self._data[-1])
		txt += "FEATURES: %d\n"%(len(self._features_vals))
		i = 1
		for vals in self._features_vals:
			txt += " [%02d]: %s\n"%(i,vals)
			i+=1
		txt += "------------------------------------------------------------\n"
		return txt

	"""
	Returns true if has been parsed

	@return true / false depending on if has been parsed
	"""
	def isParsed(self):
		return self._isParsed

	"""
	Returns the number of items in the dataset (the number of rows)

	@return 	number of items (rows)
	"""
	def itemsCount(self):
		return self._rows

	"""
	@return 	number of features (cols)
	Returns the number of features in the dataset (the number of cols)

	"""
	def featuresCount(self):
		return self._cols

	"""
	Returns the dataset information as a matrix of rows and cols containing in
	each row an example to be classified / validated with all its attributes
	defined

	@return 	dataset information
	"""
	def getData(self):
		return self._data

	"""
	Returns the features that are present in the dataset in the following format
		[[feature1_val1,feature1_val2,...],...]
	The features values contain exactly what they had when they were loaded

	@return 	classes information
	"""
	def getFeatures(self):
		return self._features_vals

	"""
	Returns the features information that are in a human language
	so that we can print useful information

	@return 	features meaning
	"""
	def getFeaturesMeaning(self):
		return self._features_mean

	"""
	Removes the samples that contain unknown values
	"""
	def _removeUnknown(self):
		for sample in self._data:
			if '?' in sample:
				self._data.remove(sample)
