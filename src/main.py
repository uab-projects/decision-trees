#!/usr/bin/python
# -*- coding: utf-8 -*-
# libraries
import core.log
from cli.arguments.parsers import DEFAULT_PARSER
from cli.arguments.constants import *
from core.dataset.reader import DataReader,AttrReader
from core.dataset.dataset import *
from core.dataset.constants import *
from core.dataset.trainingset import *
from core.dataset.validationset import *
from core.algorithms.treegrowing import BasicTreeGrowingAlgorithm
from core.algorithms.ID3 import ID3Algorithm
from core.algorithms.C45 import C45Algorithm
import logging
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
Training set object to use to generate a decision tree
"""
trainingSet = None

"""
Algorithm to use to generate a decision tree
"""
algorithm = None

# functions
"""
Reads the dataset specified in the arguments and returns the reader used with
the data already loaded. If fails, exits the application

@return 	reader object
"""
def selectDataset():
	dataSet_file = os.path.join(
		DATASET_PATH,
		args.dataset,
		args.dataset+DATASET_EXT
	)
	reader = DataReader(dataSet_file)
	try:
		reader.read()
	except Exception as e:
		LOGGER.critical("Unable to read dataset file %s: %s",
			dataSet_file,str(e))
		sys.exit(1)
	return reader

"""
Returns the user selected percentage of samples to sent to the training set
or either exits the software if invalid

@return 	percentage to send to training set
"""
def getTrainingPerentage():
	pc = args.percent
	if pc < 0. or pc > 1.:
		LOGGER.critical("""Training set percentage must be a number between 0
		and 1""")
		sys.exit(1)
	return pc



"""
Selects from the arguments the algorithm to use, and creates an object with
that algorithm

@return 	algorithm object
"""
def selectAlgorithm():
	if args.algorithm == "id3":
		algorithm = ID3Algorithm(trainingSet)
	elif args.algorithm == "c4.5":
		algorithm = C45Algorithm(trainingSet)
	elif args.algorithm == "dummy":
		algorithm = BasicTreeGrowingAlgorithm(trainingSet)
	else:
		LOGGER.critical("""Algorithm %s does not exist or is not yet
		implemented""",args.algorithm)
		sys.exit(1)
	return algorithm

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
	#Switching log level
	root_logger = logging.getLogger()
	root_logger.setLevel(LOGS_LEVELS[LOGS.index(args.log_level)])
	# Welcome
	LOGGER.info("Welcome to the Decision Trees software")
	# Read dataset file
	reader = selectDataset()
	# Percentage
	trainingSet_pc = getTrainingPerentage()
	LOGGER.info("Specified percentage to sent to training set is %f",
		trainingSet_pc)
	# Create training set and validation set from file
	dataset = Dataset(reader.getData())
	sets = dataset.getSets(trainingSet_pc)
	trainingSet = TrainingSet(sets[0])
	validationSet = ValidationSet(sets[1],args.classifier)
	# Show it
	if args.show_dataset:
		LOGGER.info(trainingSet)
	# Create algorithm
	algorithm = selectAlgorithm()
	tree = algorithm(args.classifier)
	for pre, fill, node in RenderTree(tree):
		print("%s%s" % (pre, node.name))
	#accuracy = validationSet.validateTree(tree)
	# Load attribute set
	attribs = AttrReader(os.path.join(DATASET_PATH,args.dataset,args.dataset+ATTRSET_EXT))
	try:
		attribs.read()
		attribs.parse()
	except Exception as e:
		LOGGER.warning("Unable to adquire attributes for the dataset: %s",e)
	# Apply attributes
	if(attribs.isParsed()):
		trainingSet.setFeaturesMeaning(attribs.getAttr())
		validationSet.setFeaturesMeaning(attribs.getAttr())
		try:
			algorithm.translate(tree)
		except Exception as e:
			LOGGER.error("Tree was not translated: %s",e)
	# Print tree
	if args.show_tree:
		for pre, fill, node in RenderTree(tree):
			print("%s%s" % (pre, node.name))
	# Check accuracy
	#LOGGER.info("tree accuracy is: %f"%(accuracy))
