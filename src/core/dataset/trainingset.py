import numpy as np
from .dataset import *

class TrainingSet(Dataset):
	__slots__ = ["_data"]

	"""
	Initializes a new training set given a DataSet object, turning its data
	into numpy arrays for optimization purposes

	@param 		dataset 	dataset to initialize the training set
	"""
	def __init__(self, dataset):
		super().__init__(dataset)
		self._toNumpy()

	def _toNumpy(self):
		self._data = np.array([
			np.array([self._classes[j].index(self._data[i][j])
				for j in range(self._cols)],dtype=np.uint16)
					for i in range(self._rows)])
