#!/usr/bin/python
# -*- coding: utf-8 -*-
# libraries
import core.log
from cli.arguments.parsers import DEFAULT_PARSER
from cli.arguments.constants import *
from core.dataset.readers import FileDatasetReader
from core.dataset.textdataset import TextDataset
from core.dataset.splitter import DatasetSplitter
from core.dataset.filters import DatasetFilterer
from core.dataset.constants import *
from core.dataset.trainingset import *
from core.dataset.validationset import *
#from core.measure.confusionmatrix import ConfusionMatrix
from core.algorithms.treegrowing import BasicTreeGrowingAlgorithm
from core.algorithms.ID3 import ID3Algorithm
from core.algorithms.C45 import C45Algorithm
from core.helpers.types import *
from anytree.dotexport import RenderTreeGraph
import logging
import pickle
import os
import platform
from anytree import *
import sys

# constants
LOGGER = logging.getLogger(__name__)

# variables
"""
Arguments namespace
"""
args = None

"""
Dataset with both training and validation data, useful to guess the feature
values
"""
wholeDataset = None

"""
Data extracted from the training source
"""
trainingData = None

"""
Data extracted from the validation source
"""
validationData = None

"""
Training set(s) and validation set(s) object(s) to use to generate a decision tree
"""
datasets = None

# functions
"""
Loads the reader object with the proper reader and reads the dataset, saving it
into the wholeDataset variable according to the dataset specified in the arguments
"""
def readDataset():
	# globals
	global trainingData, validationData, wholeDataset
	# dataset is from a file
	source = os.path.join(DATASET_PATH,args.dataset)
	datasetReader = FileDatasetReader(source)
	try:
		datasetReader.read()
	except Exception as e:
		LOGGER.critical("Unable to read dataset from folder %s: %s",
			source,str(e))
		sys.exit(1)

	# obtain samples data
	trainingData = datasetReader.getTrainingData()
	validationData = datasetReader.getValidationData()

	# check at least we have training data
	if trainingData is None:
		LOGGER.critical("No training data was loaded for the dataset %s. We can't continue",source)
		sys.exit(1)
	if validationData is None:
		validationData = []

	# Filter
	if args.filter != "none":
		trainingFilterer = DatasetFilterer(trainingData)
		validationFilterer = DatasetFilterer(validationData)
		# unknown rows
		if args.filter == "remove-unknown-rows":
			LOGGER.info("Applying filter to delete rows with unknown values")
			trainingFilterer.deleteUnknownSamples()
			validationFilterer.deleteUnknownSamples()
		# update data
		trainingData = trainingFilterer.getData()
		validationData = validationFilterer.getData()

	# Reading result
	wholeData = trainingData + validationData
	LOGGER.info("Loaded training%s data from file, total %d samples with %d features"," and validation" if validationData is not None else "",len(wholeData),len(wholeData[0]))
	wholeDataset = TextDataset(wholeData)
	# check if we have features
	if datasetReader.getFeaturesData() == None:
		LOGGER.warning("Dataset has no features meaning set, the tree will may be not very useful")
	else:
		wholeDataset.setFeaturesMeaning(datasetReader.getFeaturesData())
	# calculate features
	features = wholeDataset.getNumericFeatures()
	if args.show_features:
		for feature in features:
			print(feature,end='')

"""
Checks if there's any validation data and generates the training and validation numerical datasets. If not, switches the splitting meta-algorithm and generates the trainining sets and validation sets necessary as specified by the algorithm
"""
def generateDatasets():
	global datasets
	if validationData is None or not len(validationData):
		# switch algorithm and generate sets
		LOGGER.info("No validation set found, using splitting algorithm")
		splitter = DatasetSplitter(wholeDataset)
		if args.splitter == "holdout":
			datasets = [splitter.holdout(getHoldoutPercentage())]
			LOGGER.info("Applied holdout splitting: created training set has %d elements, validation set has %d elements from a total of %d ",
				datasets[0][0].getSampleCount(),
				datasets[0][1].getSampleCount(),
				wholeDataset.getSampleCount())
		else:
			LOGGER.critical("The splitting method %s has not been implemented yet, sorry :(",args.splitter)
			sys.exit(1)
	else:
		# validation set is defined
		datasets = [(
			wholeDataset.getNumericDataset(trainingData,TrainingSet),
			wholeDataset.getNumericDataset(validationData,ValidationSet)
		)]

"""
Returns the user selected percentage of samples to sent to the training set or either exits the software if invalid, when using holdout splitting method

@return 	percentage to send to training set
"""
def getHoldoutPercentage():
	pc = args.percent
	if pc < 0. or pc > 1.:
		LOGGER.critical("""Holdout splitting method training set percentage must be a number between 0 and 1. You have specified %f""",pc)
		sys.exit(1)
	return pc

"""
Return the user selected number of groups when performing cross-validation splitting algorithm. If not valid, exits the application

@return 	number of groups to set to the cross-validation
"""
def getCrossValidationK():
	k = args.k
	if k < CROSSVALID_K_MIN or k > wholeDataset.getSampleCount():
		LOGGER.critical("""Cross-validation splitting method number of groups (k) specified (%d) has to be minimum %d and maximum %d (because is the size of the dataset)""",k,CROSSVALID_K_MIN, wholeDataset.getSampleCount())
		sys.exit(1)
	return k


"""
Selects from the arguments the algorithm to use, and creates an object with
that algorithm

@return 	algorithm object
"""
def selectAlgorithm():
	if args.algorithm == "id3":
		algorithm = ID3Algorithm
	elif args.algorithm == "c4.5":
		algorithm = C45Algorithm
	elif args.algorithm == "dummy":
		algorithm = BasicTreeGrowingAlgorithm
	else:
		LOGGER.critical("""Algorithm %s does not exist or is not yet
		implemented""",args.algorithm)
		sys.exit(1)
	return algorithm

"""
Selects the classifier to generate the decision trees by setting this classifier as the target. Converts to integer if string is specified and features are available
"""
def selectClassifier():
	target = args.classifier
	if isNatural(target):
		return target
	else:
		feature_names = wholeDataset.getFeaturesNames()
		if feature_names == None:
			# unable to select from features
			LOGGER.critical("Unable to select classifier column, the classifier %s is not a natural number and no feature names are available for the dataset to select it using column names", target)
			sys.exit(1)
		else:
			# try to select from features
			try:
				target = feature_names.index(target)
			except ValueError:
				LOGGER.info("Selected classifier %s does not exist. Available classifiers for this dataset are: %s", target, feature_names)
				sys.exit(1)
	return target

"""
Takes the system arguments vector and tries to parse the arguments in it given
the argument parser specified and returns the namespace generated

@param 	parser 	the ArgumentParser objects to use to parse the arguments
@param 	args 	arguments to parse (default is sys.argv)
@return namespace of parsed arguments
"""
def parseArguments(parser, args=None):
	return parser.parse_args(args)

if __name__ == "__main__":
	# Prepare coding
	if platform.system() == "Windows":
		os.system("chcp 65001")
	args = parseArguments(DEFAULT_PARSER)
	# Switching log level
	root_logger = logging.getLogger()
	root_logger.setLevel(LOGS_LEVELS[LOGS.index(args.log_level)])
	# Welcome
	LOGGER.info("Welcome to the Decision Trees software")
	# Read dataset from sources
	readDataset()
	# Show dataset information
	if args.show_dataset:
		LOGGER.info(wholeDataset)
	# Generate training sets and validation sets
	generateDatasets()
	# Create algorithm
	algorithm = selectAlgorithm()
	classifier = selectClassifier()
	#confusionMatrix = ConfusionMatrix(wholeDataset.getFeaturesNumValues()[classifier])
	LOGGER.info("Starting to generate decision tree(s)")
	# Loop datasets and perform classifications
	trainingSet, validationSet = datasets[0][0], datasets[0][1]
	if args.from_cache:
		# Trees are cached
		try:
			tree = pickle.load(open(TREE_CACHE_FILE, "rb"))
		except Exception as e:
			LOGGER.critical("Unable to load tree from cache. Make sure cache file %s exists and is readable (%s)",TREE_CACHE_FILE,e)
	else:
		# Trees have to be calculated
		for dataset in datasets:
			trainingSet, validationSet = dataset[0], dataset[1]
			tree = algorithm(trainingSet)(classifier)
	# Give general accuracy information
	#print(confusionMatrix)
	LOGGER.info("Tree generated")
	if args.enable_cache and not args.from_cache:
		LOGGER.info("Caching generated trees into a file...")
		try:
			pickle.dump(tree, open(TREE_CACHE_FILE, "wb"))
		except:
			LOGGER.warning("Unable to cache generated trees")
	# Validate accuracies
	accuracy = validationSet.validateTree(tree, classifier)
	LOGGER.info("accuracy %s"%accuracy)
	# Print tree
	if args.show_tree:
		LOGGER.info("Attempting to translate the tree for better comprehension")
		algorithm.translate(tree,classifier, wholeDataset)
		LOGGER.info("Tree translated. Enjoy ;)")
		for pre, fill, node in RenderTree(tree):
			print("%s%s" % (pre, node.meaning))
		if len(datasets) > 1:
			LOGGER.warning("The tree provided is the last calculated, you are using a splitting method that generates more than 1 tree, so this tree does not guarantee the provided measures")
	# Save tree
	if args.output is not None:
		try:
			RenderTreeGraph(tree).to_picture(args.output)
		except Exception as e:
			LOGGER.error("Unable to export to file %s. Exception occurred. Check you have GraphicViz installed and the file is writable or does not exist yet. (%s)",args.output,e)
