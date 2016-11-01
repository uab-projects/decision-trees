from .treegrowing import *
import math

#Basic ID3 algorithm version
class ID3Algorithm(BasicTreeGrowingAlgorithm):
	def _splitCriterion(self, trainingSet, featureSet):
		#create count-list for every attribute remaining in featureSet
		entropy_general = self._H(self._countTargetClasses(trainingSet))
		entropy_list = self._entropyOfFeatures(trainingSet, featureSet)

		#return the attribute with the maximum gain (minimum entropy) if any
		gain_list = self._gain(entropy_list, entropy_general)
		feature = gain_list.index(max(gain_list))
		return featureSet[feature]

	"""
	Returns the list of entropies of the possible attributes to classify
	as the next node of the tree
	"""
	def _entropyOfFeatures(self, trainingSet, featureSet):
		entropy_list = []
		for feature in featureSet:
			#calculate entropy for the next assigment
			if self._continuous[feature]:
				print("Evaluating feature %d"%feature)
				print(self._entropyOfContinuousFeature(trainingSet, feature))
				print(self._thresholds[feature])
			#else:
			entropy_list.append(self._entropyOfFeature(trainingSet,feature,self._features[feature],self._data[:,feature]))
		return entropy_list

	"""
	Calculates the entropy for a specific feature
	"""
	def _entropyOfFeature(self, trainingSet, feature, feature_values, feature_data):
		acc = 0.0
		for value in feature_values:
			new_ts = trainingSet * np.where(feature_data == value,True,False)
			tr = self._countTargetClasses(new_ts)
			acc += np.sum(new_ts)/np.sum(trainingSet) * self._H(tr)
		return acc

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

	"""
	Given a feature, and the current training set we're dealing with, calculates the entropy for that continuous feature and updates the threshold setting it to the threshold whose partition has the minimum entropy

	@param 	feature			continous feature to calculate entropy
	@param 	trainingSet		current filtered samples we're dealing with
	@return the lowest entropy available for the partitioning of this feature
	"""
	def _entropyOfContinuousFeature(self, trainingSet, feature):
		feature_values = [0,1]
		# loop all possible values for the feature
		values = self._features[feature]
		entropy_min,best_threshold = 99, None
		for i in range(len(values)-1):
			threshold = (values[i]+values[i+1])/2.
			# create partition according to threshold
			feature_discrete = np.where(self._data[:,feature] < threshold, 0, 1)
			entropy = self._entropyOfFeature(trainingSet, feature, feature_values, feature_discrete)
			# check if there was an improvement
			if entropy < entropy_min:
				entropy_min = entropy
				best_threshold = threshold
		self._thresholds[feature] = best_threshold
		return entropy_min
