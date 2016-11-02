# libraries
import logging
import numpy as np

# constants
LOGGER = logging.getLogger(__name__)

"""
Confusion matrix with true/false positives and negatives that will able to generate different metrics to measure the efficiency of one or more decision trees
"""
class ConfusionMatrix(object):
	"""
	@attr   _matrix	 	confusion matrix containing the hits and misses
	@attr   _values	 	values the target feature can take
	@attr 	_names 		meaning for that values
	"""
	__slots__ = ["_matrix","_values","_names"]

	def __init__(self, values, names = None):
		self._values = values
		self._names = names
		self._matrix = np.zeros((len(values),len(values)),dtype=np.uint32)
		if names == None:
			self._names = list(map(str,self._values))

	"""
	Given the result obtained from classifying a sample using a decision tree and the supposed result for that sample, adds a new value to the correct matrix cell

	@param 	result 			result obtained from classifying
	@param 	supposedResult	result that is the truth (according to Plato)
	"""
	def addResult(self, result, supposedResult):
		self._matrix[result][supposedResult] += 1

	"""
	Combines the current confusion matrix with another confusion matrix to be more confused at the end
	"""
	def __add__(self, other):
		self._matrix += other.getMatrix()

	"""
	Prints a representation of the confusion matrix
	"""
	def __str__(self):
		LOGGER.warning("I'm confused, the matrix could be wrong or crazy because alcoholic illnesses")
		return str(self._matrix)

	"""
	Returns the real matrix
	"""
	def getMatrix(self):
		return self._matrix
