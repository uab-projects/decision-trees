from .ID3 import *
import math

class C45Algorithm(ID3Algorithm):
	def _splitCriterion(self, trainingSet, attributeSet):
		#create count-list for every attribute remaining in attributeSet
		entropy_general = self._H(self._countTargetClasses(trainingSet))
		entropy_list = self._entropyOfAttributes(trainingSet, attributeSet)

		#return the attribute with the maximum gain (minimum entropy) if any
		gain_list = self._gain(entropy_list, entropy_general)
		split_list = self._splitInfoOfAttributes(trainingSet, attributeSet)
		gain_ratio = C45Algorithm._gainRatio(gain_list, split_list)
		return attributeSet[gain_ratio.index(max(gain_ratio))]

	def _gainRatio(gain_list,split_list):
		return [gain_list[i]/split_list[i] if split_list[i] else 0.
			for i in range(len(gain_list))]

	"""
	Returns the list of entropies of the possible attributes to classify
	as the next node of the tree
	"""
	def _splitInfoOfAttributes(self, trainingSet, attributeSet):
		splitInfo_list = []
		card_s = np.sum(trainingSet)
		for attr in attributeSet:
			#calculate entropy for the next assigment
			acc = 0.0
			for value in range(len(self._trainingClasses[attr])):
				s_v = trainingSet * self._filterByAttributeValue(attr,value)
				card_s_v = np.sum(s_v)
				x = card_s_v / card_s
				if x:
					acc += x * math.log(x,2)
			splitInfo_list.append(-acc)
		return splitInfo_list
