from .ID3 import *
import math

class C45Algorithm(ID3Algorithm):
	def _splitCriterion(self, trainingSet, featureSet):
		#create count-list for every feature remaining in featureSet
		entropy_general = self._H(self._countTargetClasses(trainingSet))
		featureDatas, featureValues, featureEntropies, featureThresholds = self._calculateFeatures(trainingSet, featureSet)

		#return the attribute with the maximum gain (minimum entropy) if any
		gain_list = self._gain(featureEntropies, entropy_general)
		split_list = self._splitInfoOfFeatures(trainingSet, featureSet, featureValues, featureDatas)
		gain_ratio = C45Algorithm._gainRatio(gain_list, split_list)
		feature_index = gain_ratio.index(max(gain_ratio))
		feature = featureSet[feature_index]

		# Create node
		featureNode = Node(feature)
		featureNode.featureValues = featureValues[feature_index]
		featureNode.featureData = featureDatas[feature_index]
		featureNode.isContinuous = self._continuous[feature]
		featureNode.threshold = featureThresholds[feature_index]
		return featureNode

	def _gainRatio(gain_list,split_list):
		return [gain_list[i]/split_list[i] if split_list[i] else 0.
			for i in range(len(gain_list))]

	"""
	Returns the list of entropies of the possible features to classify
	as the next node of the tree
	"""
	def _splitInfoOfFeatures(self, trainingSet, featureSet, featureValues, featureDatas):
		splitInfo_list = []
		card_s = np.sum(trainingSet)
		for feature_index in range(len(featureSet)):
			#calculate entropy for the next assigment
			acc = 0.0
			feature = featureSet[feature_index]
			for value in featureValues[feature_index]:
				s_v = trainingSet * self._filterByFeatureValue(feature,value,featureDatas[feature_index])
				card_s_v = np.sum(s_v)
				x = card_s_v / card_s
				if x:
					acc += x * math.log(x,2)
			splitInfo_list.append(-acc)
		return splitInfo_list
