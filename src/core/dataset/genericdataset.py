# libraries
import numpy as np

"""
Contains a set of samples that will be used to run a decision tree algorithm
to generate a classifying tree in order to classify them. The samples are represented as a matrix where each row is a sample and each column of a row sets the feature values for the sample
"""
class GenericDataset(object):
	"""
	@attr   _data		   sample data, as a matrix of rows that represent samples and cols that set its properties
	@attr   _n_samples	  number of samples in the data set (rows of the data)
	@attr   _n_features	 number of features for each sample (columns of the data)
	@attr   _features_vals  list containing for each feature, the possible values in it
	@attr 	_continuous 	list containing per each column a true value if is a continuous feature values or not
	"""
	__slots__ = ["_data","_n_samples","_n_features","_features_vals","_continuous"]

	"""
	Initializes a new data set with the data given, setting the number of items and features manually, or either being automatic (then len of data are rows and len of data[0] will be cols)

	@param  data			the data of the datset
	@param  features_vals   values suitable for each feature, specified as a list:
		[[feature1_val1,feature1_val2,..],...]
	@param 	continuous 	list specifying the continuous attributes
	@param  samples		 number of samples in the dataset (default is calculated automatically)
	@param  features		number of features per sample (default is calculated automatically using first sample in the data)
	"""
	def __init__(self, data, features_vals = None, continuous = None, samples = None, features = None):
		self._data = data
		self._continuous = continuous
		self._features_vals = features_vals
		self._n_samples = samples
		self._n_features = features
		# Automatic calculations
		if samples == None:
			self._n_samples = len(data)
		if features == None:
			self._n_features = len(data[0]) if len(data) else 0
		if features_vals == None:
			self._guessFeaturesValues()
		if continuous is None:
			self._continuous = np.zeros(self._n_samples,dtype=bool)

	"""
	Loops through all samples to learn all the possible values that each feature can have. This calculation is only necessary if no features values have been specified

	After the calculus, the list of features and its possible values, _features_vals is saved
	"""
	def _guessFeaturesValues(self):
		# Recognize features and its values
		self._features_vals = [sorted(list(set([sample[feature] for sample in self._data])))
			for feature in range(self._n_features)]

	"""
	Returns the data of the dataset

	@return	 data matrix of samples and its features
	"""
	def getData(self):
		return self._data

	"""
	Returns the list of possible values for each feature

	@return	 list of possible values per feature
	"""
	def getFeaturesValues(self):
		return self._features_vals

	"""
	Returns the list of possible values for the feature specified

	@param	  feature	 the feature to get valid values (specified as a number, where the number is the column of the feature)
	"""
	def getFeatureValues(self, feature):
		assert feature >= 0 and feature < self._n_features, """cannot retrieve values for a feature: specified feature %d does not exist"""%feature
		return self._features_vals[feature]


	"""
	Returns the number of samples in the dataset

	@return	 number of samples in the dataset
	"""
	def getSampleCount(self):
		return self._n_samples

	"""
	Returns the number of features in the dataset for each sample

	@return	 number of features for each sample in the dataset
	"""
	def getFeaturesCount(self):
		return self._n_features

	"""
	Returns a list containing per each feature, a true or false value, depending if the feature values of that feature are continuous or not

	@return 	list of features whose values are continuous
	"""
	def getContinuousFeatures(self):
		return self._continuous

	"""
	Returns a text containing the basic information for the dataset: number of samples in it, features, per sample and the first and last sample. It also returns the possible values for each feature (if not disabled in the argument)

	@param  features_vals   true to print features values
	"""
	def __str__(self, features_vals = True):
		txt =  "%s Specifications\n"%self.__class__.__name__
		txt += "------------------------------------------------------------\n"
		txt += "SIZE:	 %d samples, %d features per sample\n"%(self._n_samples,self._n_features)
		txt += "HEAD:	 %s\n"%(self._data[0])
		txt += "TAIL:	 %s\n"%(self._data[-1])
		if features_vals:
			txt += "FEATURES: List of features and possible values\n"
			i = 1
			for vals in self._features_vals:
				txt += " [%02d]: %s\n"%(i,vals)
				i+=1
		txt += "------------------------------------------------------------\n"
		return txt
