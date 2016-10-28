from .treegrowing import *
import math

#Basic ID3 algorithm version
class ID3Algorithm(BasicTreeGrowingAlgorithm):
	def _splitCriterion(self, trainingSet, attributeSet):
		#create count-list for every attribute remaining in attributeSet
		entropy_general = self._H(self._countTargetClasses(trainingSet))
		entropy_list = self._entropyOfAttributes(trainingSet, attributeSet)

		#return the attribute with the maximum gain (minimum entropy) if any
		gain_list = self._gain(entropy_list, entropy_general)
		return attributeSet[gain_list.index(max(gain_list))]

	"""
	Returns the list of entropies of the possible attributes to classify
	as the next node of the tree
	"""
	def _entropyOfAttributes(self, trainingSet, attributeSet):
		entropy_list = []
		for attr in attributeSet:
			#calculate entropy for the next assigment
			acc = 0.0
			for value in range(len(self._trainingClasses[attr])):
				new_ts = trainingSet * self._filterByAttributeValue(attr,value)
				tr = self._countTargetClasses(new_ts)
				acc += np.sum(new_ts)/np.sum(trainingSet) * self._H(tr)

			entropy_list.append(acc)
		return entropy_list

	"""
	Returns the gain given a list of entropies and the general entropy

	@param 	general_entropy		entropy of the current classification
	@param 	entropies	 		list of entropies if we classify as some
								attribute
	@return gain for each entropy
	"""
	def _gain(self, entropies,general_entropy):
		return list(map(lambda e: general_entropy-e,entropies))

	"""
	Calculates and returns the entropy of the given values taken as a list

	@param  amounts	 list of the amounts of elements for one attribute
						belonging the target class
	"""
	def _H(self,amounts):
		acc=0
		total_classes = sum(amounts)

		for el in amounts:
			if el != 0:
				prob = el/total_classes
				acc += prob * math.log(prob,2)

		return -acc

	"""
	Selects a percentatge from the original sample in order to be the training
	set and the other percentatge will be the testSet

	@param  percent percentatge of the sample wanted to belong to training
					must be a value between 0 and 1
	"""
	def createTest(percent):
		if (percent < 1 and percent >0):
			elements_count = int(self._cols*percent)
			#select elements_count from self._trainingSet
			#select total-(elements_count) elements from self._trainingSet and allocate in self._testSet
		else:
			#do nothing
			return None
