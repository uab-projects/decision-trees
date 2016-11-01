from .genericdataset import *
from .constants import *
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

class TextDataset(GenericDataset):
	"""
	@attr 	_features_mean		meaning of each feature and of each feature value in the dataset. The format is the following one:
		[[feature1_meaning,{feature1_vals_meaning}]]
	Where feature(i)_vals_meaning is a dictionary that maps the feature value to a meaningful value.
	@attr 	_features_map_txt 	maps each text feature value to a numerical value, to then be able to extract numerical datasets
	@attr 	_features_map_num 	maps each numerical feature value to a text value (viceversa of previous attribute)
	@attr 	_features_vals_np 	numpy features valid values
	"""
	__slots__ = ["_features_map_txt","_features_map_num","_features_mean","_features_mean_np"]

	"""
	Initializes a dataset given the matrix of samples where rows are the samples and columns are the feature values for each sample

	@param 	data 	data as matrix containing samples in text format
	"""
	def __init__(self,data):
		super().__init__(data)
		self._features_mean = None
		self.__createNumericMappers()


	"""
	Generates a numerical mapper from each feature text value into a numerical value and vice versa to enable numerical dataset extraction
	"""
	def __createNumericMappers(self):
		# text to number mappers
		self._features_map_txt = [dict() for _ in range(len(self._features_vals))]
		# number to text mappers
		self._features_map_num = [
			["" for _ in range(len(feature))]
				for feature in self._features_vals]
		# assign maps
		for feature_index in range(len(self._features_vals)):
			vals = self._features_vals[feature_index]
			text_mapper = self._features_map_txt[feature_index]
			num_mapper = self._features_map_num[feature_index]
			i = 0
			for val in vals:
				text_mapper[val] = i
				num_mapper[i] = val
				i+=1

	"""
	Update the numeric mappers
	"""
	def udpateNumericMappers(self):
		self.__createNumericMappers()

	"""
	Returns a numerical, NumPy dataset object (the same object, but with numpy data) where each text feature is mapped to a number. If some data is specified rather than the data in the own dataset, those data will be used

	@return 	generic dataset with the same data and attributes but with the data mapped to numbers
	"""
	def getNumericDataset(self, data=None):
		if data == None:
			data = self._data
		num_data = np.array([
			np.array([
			int(data[sample][feature]) if self._continuous[feature] else self._features_map_txt[feature][data[sample][feature]]
				for feature in range(self._n_features)],dtype=np.uint16)
					for sample in range(self._n_samples)])
		return GenericDataset(num_data, self.getFeaturesNumValues(), self._continuous)

	"""
	Returns the list of possible values for each feature when using numerical converted dataset

	@return	 list of possible values per feature
	"""
	def getFeaturesNumValues(self):
		return [ list(map(int,self._features_vals[feature])) if self._continuous[feature] else list(range(len(self._features_vals[feature]))) for feature in range(len(self._features_vals))]

	"""
	Given a list with the meaning for each feature and each feature's values, saves it in the dataset for future operations. The list is formatted according to the following format:
		[[feature1_meaning,{feature1_dictionary}],...]
	Where feature1_dictionary maps the text in the dataset with a meaningful
	value

	@param 	features_meanings 	meaning of the features and its values
	"""
	def setFeaturesMeaning(self, features_meanings):
		self._features_mean = features_meanings
		# check if continuous values
		for feature in range(len(self._features_mean)):
			feature_mean = self._features_mean[feature]
			if DATASET_FEATURE_CONT in feature_mean[1].values():
				self._continuous[feature] = True

	"""
	Returns the features information that are in a human language
	so that we can print useful information

	@return 	features meaning
	"""
	def getFeaturesMeaning(self):
		return self._features_mean

	"""
	Returns the names for each feature, in a string list. The list is ordered so the first string matches the first feature name and so on

	@return 	feature name list (if features are set, otherwise none)
	"""
	def getFeaturesNames(self):
		return [mean[0] for mean in self._features_mean] if self._features_mean != None else None

	"""
	Given a feature, returns its meaning, as appears in the features meaning
	dictionary, or returns the same feature if no meaning has been found

	@param 	feature 	the feature number (as ordered in the dataset column)
	@return feature name
	"""
	def getFeatureMeaning(self, feature):
		assert feature >= 0 and feature < self._n_features, """cannot retrieve meaning for a feature: specified feature %d does not exist"""%feature
		return self._features_mean[feature][0] if self._features_mean != None else "Feature %d"%feature

	"""
	Given a feature and a value for that feature, returns its meaning if exists
	or returns the same value if no meaning found

	@param 	feature 	the feature number (as ordered in the dataset column)
	@param  value 		the value for that feature (as appears in the dataset
	samples)
	@return feature value name, or its representation string if no meaning available
	"""
	def getFeatureValueMeaning(self, feature, value):
		assert feature >= 0 and feature < self._n_features, """cannot retrieve meaning for a feature value: specified feature %d does not exist"""%feature
		feature_values = self._features_vals[feature]
		assert value in feature_values, """cannot retrieve meaning of value %s of the feature %s, the feature value does not exist"""%(feature, value)
		return self._features_mean[feature][1][value] if self._features_mean != None else "Feature %d = %s"%(feature,value)

	"""
	Given a feature and a numerical value for that feature, returns its meaning
	if exists or returns the same value if no meaning found

	@param 	feature 	the feature number (as ordered in the dataset column)
	@param 	value 		the numerical index for that value as mapped to numerical data sets
	@return feature value name, or its representation string with short name if
	no meaning is available
	"""
	def getFeatureNumValueMeaning(self, feature, value_num):
		value = self._features_map_num[feature][value_num]
		return self.getFeatureValueMeaning(feature, value)

	"""
	Given a feature and and a feature value, returns the feature value numerical value map.

	@param 	feature 	the feature number (as ordered in the dataset column)
	@param 	value 		the feature value to map to number
	@return feature value number or None if feature value does not exist
	"""
	def getFeatureValueMap(self, feature, value):
		assert feature >= 0 and feature < self._n_features, """cannot retrieve mapping of a value for a feature: specified feature %d does not exist"""%feature
		value = None
		try:
			value = self._features_map_txt[feature][value_num]
		except KeyError:
			LOGGER.warning("Feature %d, value %d has no text map",feature, value)
		return value

	"""
	Returns the information about this dataset, that is basically every dataset information, but adding the meaning for the features if it has been specified
	"""
	def __str__(self):
		txt = super().__str__(self._features_mean == None)
		if self._features_mean == None:
			txt += "Features meaning has not been specified yet\n"
		else:
			txt += "FEATURES: List of features and possible values\n"
			for feature in range(len(self._features_vals)):
				meaning = self._features_mean[feature]
				txt += " [%s]: %s\n"%(meaning[0],list(meaning[1].values()) if not self._continuous[feature] else "<continuous>")
		txt += "------------------------------------------------------------\n"
		return txt
