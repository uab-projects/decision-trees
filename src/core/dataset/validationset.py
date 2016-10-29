import numpy as np
from .dataset import *

class ValidationSet(Dataset):
	__slots__ = ["_data", "_target"]

	"""
	Initializes a new validation set given a DataSet object and the percentatge
	of the DataSer, turning its data into numpy arrays for optimization purposes

	@param 		dataset 	dataset to initialize the training set
	@param		target		target value to evaluate samples
	"""
	def __init__(self, dataset, target):
		super().__init__(dataset)
		self._toNumpy()
		self._target = target

	def _toNumpy(self):
		self._data = np.array([
			np.array([self._classes[j].index(self._data[i][j])
				for j in range(self._cols)],dtype=np.uint16)
					for i in range(self._rows)])
	"""
	Checks the tree quality by using a part of the DataSet as Validation Set and
	calculating the hits and misses on it
	"""
	def validateTree(self, tree):
		hits = 0.
		for sample in self._data:
			print(sample)
			if self._validateSample(tree, sample):
				hits += 1
		print(hits)
		return hits/self._rows

	def _validateSample(self, tree, sample):

		node = tree
		while not node.is_leaf:
			childs = [e.name for e in node.children]
			# has children -> evalueate next node with next attribute
			attribute = sample[node.name]
			print(childs,attribute)
			next_node = childs.index(attribute)
			node = node.children[next_node].children[0]

		# is leaf -> evaluate target
		values = self._classes[self._target]

		if values.index(node.name) == sample[self._target]:
			print("HIT!")
			return True

		return False
