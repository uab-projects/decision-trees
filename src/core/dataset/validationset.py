import numpy as np
#from core.measure.confusionmatrix import ConfusionMatrix
from .genericdataset import *

class ValidationSet(GenericDataset):
	"""
	Initializes a new validation set given a DataSet object and the percentatge
	of the DataSer, turning its data into numpy arrays for optimization purposes

	@param 		dataset 	dataset to initialize the training set
	@param		target		target value to evaluate samples
	"""
	#def __init__(self, dataset):
	#	super().__init__(dataset)
	#	self._target = target

	"""
	Checks the tree quality by using a part of the DataSet as Validation Set and
	calculating the hits and misses on it
	"""
	def validateTree(self, tree, target):
		#confusionMatrix = ConfusionMatrix(self._features_vals[target])
		hits = 0.
		for sample in self._data:
			hits += self._classifySample(tree, sample, target)
			#confusionMatrix.addResult(result, sample[target])
		return hits/self._n_samples

	def _classifySample(self, tree, sample, target):
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
		values = self._features_vals[target]

		if values.index(node.name) == sample[target]:
			return True
		return False
