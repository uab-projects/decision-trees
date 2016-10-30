from .genericdataset import *
import logging

LOGGER = logging.getLogger(__name__)

class TextDataset(GenericDataSet):
	"""
	@attr 	_features_map_txt 	maps each text feature value to a numerical value, to then be able to extract numerical datasets
	@attr 	_features_map_num 	maps each numerical feature value to a text value (viceversa of previous attribute)
	@attr 	_features_mean		meaning of each feature and of each feature value in the dataset. The format is the following one:
		[[feature1_meaning,{feature1_vals_meaning}]]
	Where feature(i)_vals_meaning is a dictionary that maps the feature value to a meaningful value.
	@attr 	_features_mean_np 	meaning of each feature value when using a numerical (NumPy) dataset extracted from this dataset
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
		self._features_mean_np = None
		self.__createNumericMappers()


	"""
	Generates a numerical mapper from each feature text value into a numerical value and vice versa to enable numerical dataset extraction
	"""
	def __createNumericMappers(self):
		# text to number mappers
		self._features_map_txt = [dict() for _ in range(self._features_vals)]
		# number to text mappers
		self._features_map_num = [
			["" for _ in range(len(feature))]
				for feature in self._features_vals]
		# assign maps
		for feature_index in range(self._features_vals):
			vals = self._features_vals[feature_index]
			text_mapper = self._features_map_txt[feature_index]
			num_mapper = self._features_map_num[feature_index]
			i = 0
			for val in vals:
				text_mapper[val] = i
				num_mapper[i] = val
				i+=1

	"""
	Creates a lists of dictionaries, mapping the numerical value of each feature to the long meaning of it
	"""
	def __createNumericMeanings(self):
		self._features_mean_np = [dict() for _ in range(self._features_vals)]
		for feature_index in range(self._features_vals):
			meaning_mapper = self._features_mean[feature_index][1]
			num_meaning_mapper = self._features_mean_np[feature_index]
			text_mapper = self._features_map_txt[features_index]
			for feature_val in self._features_vals[feature_index]:
				num_meaning_mapper[text_mapper[feature_val]] = meaning_mapper[feature_val]

	"""
	Returns a numerical, NumPy dataset object (the same object, but with numpy data) where each text feature is mapped to a number

	@return 	generic dataset with the same data and attributes but with the data mapped to numbers
	"""
	def getNumericDataset(self):
		np_data = np.array([
			np.array([self._features_vals[j].index(self._data[i][j])
				for j in range(self._cols)],dtype=np.uint16)
					for i in range(self._rows)])

	"""
	Given a list with the meaning for each feature and each feature's values, saves it in the dataset for future operations. The list is formatted according to the following format:
		[[feature1_meaning,{feature1_dictionary}],...]
	Where feature1_dictionary maps the text in the dataset with a meaningful
	value

	@param 	features_meanings 	meaning of the features and its values
	"""
	def setFeaturesMeaning(self, features_meanings):
		self._features_mean = features_meanings
		self.__createNumericMeanings()

	"""
	Returns the features information that are in a human language
	so that we can print useful information

	@return 	features meaning
	"""
	def getFeaturesMeaning(self):
		return self._features_mean

	"""
	Returns the information about this dataset, that is basically every dataset information, but adding the meaning for the features if it has been specified
	"""
	def __str__(self):
		txt = super().__str__(self._features_mean == None)
		if self._features_mean == None:
			txt += "Features meanining has not been specified yet"
		else:
			txt += "FEATURES: List of features and possible values\n"
			for feature in self._features_vals:
				meaning = self._features_mean[feature]
				txt += " [%s]: %s\n"%(meaning[0],list(meaning[1].values()))
		txt += "------------------------------------------------------------\n"
		return txt
