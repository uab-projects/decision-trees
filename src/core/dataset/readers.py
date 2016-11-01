# App modules
from .constants import *

# External modules
import csv
import logging
import os
from abc import ABCMeta, abstractmethod

# Constants
LOGGER = logging.getLogger(__name__)

"""
Defines an interface to show how datasets readers must act
"""
class DatasetReader(object):
	"""
	@attr 	_trainingData 	matrix of data where rows are samples defined by its features specified in each column that will be used for training purposes
	@attr 	_validationData 	same as trainingData but for validation
	@attr 	_featuresData 		contains information about the features of the dataset
	"""
	__slots__= ["_trainingData", "_validationData", "_featuresData"]
	__metaclass__ = ABCMeta

	"""
	Separation to use between the feature definition and its values definitions
	"""
	FEATURE_DEF_SEP = ':'

	"""
	Separation to use between the feature values definitions
	"""
	FEATURE_VALUES_SEP = ','

	"""
	Separation to use between a feature value short name and a feature value long name
	"""
	FEATURE_VALUES_DEF_SEP = '='

	"""
	Reads the data from the given source and stores the data into the data attribute
	"""
	@abstractmethod
	def read(self):
		pass

	"""
	Returns the training data loaded (it will be also the validation data if no validation data has been found)

	@return 	training data matrix
	"""
	def getTrainingData(self):
		return self._trainingData

	"""
	Returns the validation data loaded (or None if no validation data has been found)

	@return 	validation data matrix
	"""
	def getValidationData(self):
		return self._validationData

	"""
	Returns information about the features present in the training and validation data in the following format
		[[feature1_meaning,feature1_values_meanings],...]
	Where feature(i)_values_meanings is a dictionary mapping the feature values of the feature i that appear in the dataset to some meaningful description

	@return 	features data if found
	"""
	def getFeaturesData(self):
		return self._featuresData

class FileDatasetReader(DatasetReader):
	"""
	@attr 	_folder 	folder where data is stored
	"""
	__slots__ = ["_folder"]

	"""
	Initializes a dataset reader from a folder source where it will try to locate the training data file and validation file

	@param 	folder 	folder where training data and validation data files should be located
	"""
	def __init__(self,folder):
		self._folder = folder

	"""
	Reads the data from the files and stores it into the attributes. If one of either two files does not exist, then will be set to None
	"""
	def read(self):
		# training file
		self._trainingData = None
		trainingFiles = list(filter(lambda x: x.endswith("." + DATASET_TRAINING_EXT), os.listdir(self._folder)))
		if len(trainingFiles):
			self._trainingData = self._readFile(os.path.join(self._folder,trainingFiles[0]))
		else:
			LOGGER.error("No training data file found in folder %s",
			self._folder)
		# validation file
		self._validationData = None
		validationFiles = list(filter(lambda x: x.endswith("." + DATASET_VALIDATION_EXT), os.listdir(self._folder)))
		if len(validationFiles):
			self._validationData = self._readFile(os.path.join(self._folder,validationFiles[0]))
		# features file
		self._featuresData = None
		featuresFiles = list(filter(lambda x: x.endswith("." + DATASET_FEATURES_EXT), os.listdir(self._folder)))
		if len(featuresFiles):
			self._featuresData = [line.rstrip('\n') for line in open(os.path.join(self._folder,featuresFiles[0]),"r")]

	"""
	Reads the data from the file and returns it as a matrix of data

	Throws IOError if fails

	@param 		filename filename to read
	@return 	data loaded or None if does not exist
	"""
	def _readFile(self, filename):
		LOGGER.debug("Starting to read %s",filename)
		data = list(csv.reader(open(filename, 'r'), delimiter=DATASET_COLS_SEP,skipinitialspace=True))
		#data = [[s.strip() for s in row] for row in data]
		LOGGER.debug("Finished reading %s", filename)
		return data

	"""
	Parses the features data to format it before returning it

	@return 	features data formatted
	"""
	def _parseFeatures(self):
		if self._featuresData == None:
			return None
		features = []
		i = 0
		for feature_definition in self._featuresData:
			# separe name and value definitions
			feature_name, feature_values = 	feature_definition.split(self.FEATURE_DEF_SEP)
			# format and check name
			feature_name = feature_name.strip()
			if not len(feature_name):
				LOGGER.warning("Feature %d has no name, setting a default name",
				i)
				feature_name = "Feature <%02d>"%(i)
			# create meaning dictionary
			meaning = (feature_name.strip(),dict())
			# parse and check feature values
			feature_values = feature_values.split(self.FEATURE_VALUES_SEP)
			if not len(feature_values):
				LOGGER.warning("Feature %s has no information about its values meanings",feature_name)
			else:
				j=0
				for value_definition in feature_values:
					# read long name and short name
					names = value_definition.split(self.FEATURE_VALUES_DEF_SEP)
					while len(names) < 2:
						names.append("")
					long_name , short_name = names[1].strip(), names[0].strip()
					# check names are correct
					if not len(short_name):
						LOGGER.error("Feature %s value definition %s is invalid: no short name (name before %s). Omitting definition",feature_name,value_definition,self.FEATURE_VALUES_DEF_SEP)
						continue
					if not len(long_name):
						LOGGER.debug("Feature %s value definition %s is invalid: no long name (name after %s). Setting default name",feature_name, value_definition, self.FEATURE_VALUES_DEF_SEP)
						long_name = short_name
					# save meaning
					meaning[1][short_name] = long_name
					j+=1
			# save feature
			features.append(meaning)
			i+=1
		return features

	"""
	Returns properly formatted features data

	@return 	features data if found, None if not
	"""
	def getFeaturesData(self):
		return self._parseFeatures()
