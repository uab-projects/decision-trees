#!/usr/bin/python
# -*- coding: utf-8 -*-
# libraries
import core.log
from cli.arguments.parsers import DEFAULT_PARSER
from core.dataset.reader import DataReader,AttrReader
from core.dataset.dataset import *
from core.dataset.constants import *
from core.dataset.trainingset import *
from core.algorithms.treegrowing import BasicTreeGrowingAlgorithm
from core.algorithms.ID3 import ID3Algorithm
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
Selects from the arguments the algorithm to use, and creates an object with
that algorithm

@return 	algorithm object
"""
def selectAlgorithm():
	if args.algorithm == "id3":
		algorithm = ID3Algorithm(trainingSet)
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
	if args.log_level=="debug":
		root_logger.setLevel(logging.DEBUG)
	elif args.log_level=="info":
		root_logger.setLevel(logging.INFO)
	elif args.log_level=="warning":
		root_logger.setLevel(logging.WARNING)
	elif args.log_level=="error":
		root_logger.setLevel(logging.ERROR)
	elif args.log_level=="critical":
		root_logger.setLevel(logging.CRITICAL)
	# Welcome
	LOGGER.info("Welcome to the Decision Trees software")
	# Read dataset file
	reader = selectDataset()
	# Create training set from file
	trainingSet = TrainingSet(reader.getData())
	# Show it
	if args.show_dataset:
		LOGGER.info(trainingSet)
	# Create algorithm
	algorithm = selectAlgorithm()
	tree = algorithm(args.classifier)
	# Load attribute set
	attribs = AttrReader(os.path.join(DATASET_PATH,args.dataset,args.dataset+ATTRSET_EXT))
	try:
		attribs.read()
		attribs.parse()
	except Exception as e:
		LOGGER.warning("Unable to adquire attributes for the dataset: %s",e)
	# Apply attributes
	if(attribs.isParsed()):
		trainingSet.applyAttributes(attribs.getAttr())
		try:
			algorithm.translate(tree)
		except Exception as e:
			LOGGER.error("Tree was not translated: %s",e)
	# Print tree
	if args.show_tree:
		for pre, fill, node in RenderTree(tree):
			print("%s%s" % (pre, node.name))
