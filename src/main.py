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
import logging
import os
import platform
from anytree import *

# constants
LOGGER = logging.getLogger(__name__)

# functions
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
	# Welcome
	LOGGER.info("Welcome to the Decision Trees software")
	# Read dataset file
	reader = DataReader(os.path.join(DATASET_PATH,args.dataset,args.dataset+DATASET_EXT))
	try:
		reader.read()
	except Exception as e:
		LOGGER.error("Unable to read dataset file: %s",str(e))
	# Create training set from file
	trainingset = TrainingSet(reader.getData())
	# Parse dataset
	if args.show_dataset:
		LOGGER.info("Dataset information:\n%s",trainingset)
	# Create algorithm
	alg = BasicTreeGrowingAlgorithm(trainingset)
	tree = alg(0)
	# Load attribute set
	attribs = AttrReader(os.path.join(DATASET_PATH,args.dataset,args.dataset+ATTRSET_EXT))
	try:
		attribs.read()
		attribs.parse()
	except Exception as e:
		LOGGER.warning("Unable to adquire attributes for the dataset: %s",e)
	if(attribs.isParsed()):
		trainingset.applyAttributes(attribs.getAttr())
		try:
			alg.translate(tree)
		except Exception as e:
			LOGGER.error("Tree was not translated: %s",e)
	for pre, fill, node in RenderTree(tree):
		print("%s%s" % (pre, node.name))
