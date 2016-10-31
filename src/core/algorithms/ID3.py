from .treegrowing import *
import math

#Basic ID3 algorithm version
class ID3Algorithm(BasicTreeGrowingAlgorithm):
	def _splitCriterion(self, trainingSet, featureSet):
		#create count-list for every attribute remaining in featureSet
		entropy_general = self._H(self._countTargetClasses(trainingSet))
		entropy_list = self._entropyOfAttributes(trainingSet, featureSet)

		#return the attribute with the maximum gain (minimum entropy) if any
		gain_list = self._gain(entropy_list, entropy_general)
		return featureSet[gain_list.index(max(gain_list))]

	"""
	Returns the list of entropies of the possible attributes to classify
	as the next node of the tree
	"""
	def _entropyOfAttributes(self, trainingSet, featureSet):
		entropy_list = []
		for feature in featureSet:
			#calculate entropy for the next assigment
			acc = 0.0
			for value in self._features[feature]:
				new_ts = trainingSet * self._filterByAttributeValue(feature,value)
				tr = self._countTargetClasses(new_ts)
				acc += np.sum(new_ts)/np.sum(trainingSet) * self._H(tr)

			entropy_list.append(acc)
		return entropy_list

	"""
	Returns the gain given a list of entropies and the general entropy

	@param 	general_entropy		entropy of the current classification
	@param 	entropies	 		list of entropies if we classify with some feature
	@return gain for each entropy
	"""
	def _gain(self, entropies,general_entropy):
		return list(map(lambda e: general_entropy-e,entropies))

	"""
	Calculates and returns the entropy of the given values taken as a list

	@param  amounts	 list of the amounts of elements for one feature value
						belonging the target feature
	"""
	def _H(self,amounts):
		acc=0
		total_classes = sum(amounts)
		for el in amounts:
			if el != 0:
				prob = el/total_classes
				acc += prob * math.log(prob,2)
		return -acc
