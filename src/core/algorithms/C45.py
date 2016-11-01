from .ID3 import *
import math

class C45Algorithm(ID3Algorithm):
	def _splitCriterion(self, trainingSet, featureSet):
		#create count-list for every feature remaining in featureSet
		entropy_general = self._H(self._countTargetClasses(trainingSet))
		entropy_list = self._entropyOfFeatures(trainingSet, featureSet)

		#return the attribute with the maximum gain (minimum entropy) if any
		gain_list = self._gain(entropy_list, entropy_general)
		split_list = self._splitInfoOfAttributes(trainingSet, featureSet)
		gain_ratio = C45Algorithm._gainRatio(gain_list, split_list)
		feature = featureSet[gain_ratio.index(max(gain_ratio))]

		# Create node
		featureNode = Node(feature)
		featureNode.featureValues = self._features[feature]
		featureNode.featureData = self._data[:,feature]
		featureNode.isContinuous = False
		featureNode.threshold = None
		return featureNode

	def _gainRatio(gain_list,split_list):
		return [gain_list[i]/split_list[i] if split_list[i] else 0.
			for i in range(len(gain_list))]

	"""
	Returns the list of entropies of the possible features to classify
	as the next node of the tree
	"""
	def _splitInfoOfAttributes(self, trainingSet, featureSet):
		splitInfo_list = []
		card_s = np.sum(trainingSet)
		for feature in featureSet:
			#calculate entropy for the next assigment
			acc = 0.0
			for value in self._features[feature]:
				s_v = trainingSet * self._filterByFeatureValue(feature,value,self._data[:,feature])
				card_s_v = np.sum(s_v)
				x = card_s_v / card_s
				if x:
					acc += x * math.log(x,2)
			splitInfo_list.append(-acc)
		return splitInfo_list
