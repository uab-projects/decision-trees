import numpy as np
from .genericdataset import *

class ValidationSet(GenericDataset):
	pass

class ValidationAlternativeSet(GenericDataset):
	"""
	Initializes a new validation set given a DataSet object and the percentatge
	of the DataSer, turning its data into numpy arrays for optimization purposes

	@param 		dataset 	dataset to initialize the training set
	@param		target		target value to evaluate samples
	"""
	def __init__(self, dataset, target):
		super().__init__(dataset)
		self._target = target

	"""
	Checks the tree quality by using a part of the DataSet as Validation Set and
	calculating the hits and misses on it
	"""
	def validateTree(self, tree):
		hits = 0.
		for sample in self._data:
			hits += self._validateSample(tree, sample)
		return hits/self._rows

	def _validateSample(self, tree, sample):
		node = tree
		while not node.is_leaf:
			childs = [e.name for e in node.children]
			# has children -> evaluate next node with next attribute
			attribute = sample[node.name]
			if not attribute in childs:
				return False
			next_node = childs.index(attribute)
			node = node.children[next_node].children[0]
		# is leaf -> evaluate target
		values = self._features_vals[self._target]

		if values.index(node.name) == sample[self._target]:
			return True
		return False
