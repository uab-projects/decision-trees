import argparse
import ast
from .constants import *

# helping methods
def evalTF(string):
	return ast.literal_eval(string.title())

# Default parser
DEFAULT_PARSER = argparse.ArgumentParser(
	# prog = 'decision-tree.py'
	# usage = (generated by default)
	description = """Given a dataset either from local or remote origins,
	analyzes the data to generate a decision tree that will be able to classify
	the data according to the attribute specified and finally validate the
	accuracy against a training set, or trying to classify random examples""",
	epilog = "<> with ♥ in ETSE UAB by ccebrecos, davidlj95 & joel.sanz",
	add_help = True,
	allow_abbrev = True
)
DEFAULT_PARSER.add_argument("-v","--version",
	action="version",
	version="Decision-Tree classifier 0.0.0 (alpha)")
DEFAULT_PARSER.add_argument("-d","--dataset",
	action="store",
	nargs="?",
	help="""specifies the dataset to load (default is %s)"""%\
		DATASET_DEFAULT,
	type=str,
	choices=DATASETS,
	default=DATASET_DEFAULT,
)
DEFAULT_PARSER.add_argument("--show-dataset",
	metavar="true|false",
	action="store",
	nargs="?",
	help="""enables or disables printing the dataset information (%s
	by default)"""%("enabled" if SHOW_DATASET_DEFAULT else "disabled"),
	type=evalTF,
	const=True,
	default=SHOW_DATASET_DEFAULT
)
DEFAULT_PARSER.add_argument("-t",
	action="count",
	help="""records the algorithm's computation time and shows them. You can
	add levels of timings by specifying repeating argument. Default timing
	level is %d"""%TIMERS_DEFAULT,
	default = TIMERS_DEFAULT
)