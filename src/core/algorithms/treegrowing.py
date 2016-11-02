# Libraries
from abc import ABCMeta,abstractmethod,abstractproperty
from anytree import *
import numpy as np
import random
import logging

LOGGER = logging.getLogger(__name__)

"""
Abstract algorithm that generates a decision tree based on a classification
criteria to try to classify according a determined variable target new examles
given a training set

Base for the ID3, C4.5 algorithms
"""
class TreeGrowingAlgorithm(object):
	__metaclass__ = ABCMeta
	"""
	@attr 	_target 			feature to classify to
	@attr 	_data               training matrix obtained from training set containing the samples to classify
	@attr 	_features		 	possible values for each feature, specified as
	list of lists
	@attr 	_continuous 		features whose values are continuous
	@attr 	_thresholds 		thresholds that set how to discretize the continuous values in order to set 2 partitions (< threshold, >= than threshold)
	@attr	_isRunning 			controls the algorithm is not run twice
	"""
	__slots__ = ["_target","_data","_features","_isRunning"]

	"""
	Initializes the tree growing algorithm given some training data

	@param 	trainingSet 		training data object
	"""
	def __init__(self, trainingSet):
		self._target = None
		self._features = trainingSet.getFeaturesValues()
		self._data = trainingSet.getData()
		self._continuous = trainingSet.getContinuousFeatures()
		self._isRunning = False

	"""
	Starts the algorithm that generates a decision tree. Returns the tree when
	the algorithm has finished.

	@param 	target 				variable to classify
	@param 	featureSet 			features to take in consideration
	"""
	def __call__(self, target, featureSet = None):
		# not running
		assert not self._isRunning, "the algorithm is alredy running"
		self._isRunning = True
		# generate features and trainingSet
		trainingSet = np.ones(len(self._data), dtype=bool)
		if featureSet == None:
			featureSet = [i for i in range(len(self._data[0]))]
		# check correct target
		assert target in featureSet, """target classifier is not defined in
		feature set, target must be %d <= target < %d"""%(0,len(featureSet))
		self._target = target
		featureSet.remove(target)
		# classify
		LOGGER.debug("Starting to generate decision tree")
		tree = self._treeGrowing(trainingSet, featureSet)
		LOGGER.debug("Decision tree generated")
		self._isRunning = False
		return tree.children[0]

	"""
	Creates a decision tree built from a given training set to evaluate and
	classify some values in the future to determine the target value

	@param  trainingSet	 	list of boolean references of the real training set
	that specify the samples of the real training set we're dealing with
	@param  featureSet 		list of features pending to classify
	@param  target		 	feature to be determined by the tree
	@return tree
	"""
	def _treeGrowing(self, trainingSet, featureSet, parent=Node("root"),
	 	depth=0):
		LOGGER.debug("-> treeGrowing Iteration (depth=%d)",depth)
		LOGGER.debug("   Parent is: %s",str(parent.name))
		if self._stopCriterion(trainingSet, featureSet):
			# leaf
			LOGGER.debug("--> Leaf reached.")
			leaf = self._selectLeaf(trainingSet)
			LOGGER.debug("    Leaf has been tagged as %s",self._features[self._target][leaf])
			Node(self._features[self._target][leaf],parent)
		else:
			# branch
			LOGGER.debug("--> Splitting tree")
			featureNode = self._splitCriterion(trainingSet, featureSet)
			featureNode.parent = parent
			# Get data from the node
			feature = featureNode.name
			featureSet.remove(feature)
			LOGGER.debug("---> Split criterion: %d",feature)
			# temporary, just an idea -- born to help, not to work
			for featureValue in featureNode.featureValues:
				# cut training set
				filteredTrainingSet = trainingSet * self._filterByFeatureValue(feature,featureValue,featureNode.featureData)
				LOGGER.debug("----> Created node for %s",featureValue)
				LOGGER.debug("      Examples:     %s (in, out)",str(np.bincount(filteredTrainingSet,minlength=2)[::-1]))
				LOGGER.debug("      Distribution: %s (%s)",self._countTargetClasses(filteredTrainingSet),self._features[self._target])
				# Recursive call
				if(np.sum(filteredTrainingSet)):
					valueNode = Node(featureValue,featureNode)
					valueNode.samples = np.sum(filteredTrainingSet)
					self._treeGrowing(filteredTrainingSet, featureSet, valueNode,depth+1)
		return parent

	"""
	Given a feature and its value, selects all the samples whose feature specified matches the specified feature value in the feature data passed

	@param 		feature 		feature whose values will be filtered
	@param 		featureValue 	value of the feature the samples must match
	@param 		featureData 	feature data of the samples
	@return 	boolean array where trues are samples in the data that match the filter
	"""
	def _filterByFeatureValue(self, feature, featureValue, featureData):
		return np.where(featureData == featureValue,True,False)

	"""
	Given a reference to the data as a boolean arrays, takes those true samples and counts how are they distributed according to the target feature

	@param 		trainingSet 	references to the original sample data that we have to take in account to calculate the count of the target classes
	@return 	list counting the occurences of each value of the classifier target feature in the training set reference specified
	"""
	def _countTargetClasses(self, trainingSet):
		return np.bincount(self._data[:,self._target][trainingSet],minlength=len(self._features[self._target]))

	"""
	Given a traning set, returns the target feature value where most of the samples are in
	"""
	def _selectLeaf(self, trainingSet):
		return self._countTargetClasses(trainingSet).argmax()

	"""
	Adds a meaning field to each node in order to then print the node with useful information. The algorithm will try to look for meanings in the original dataset, but will retrieve default values if not possible

	@param 	tree 	tree to translate
	@param 	target 	target feature to classify
	@param 	dataset dataset to retrieve meanings from
	"""
	def translate(tree, target, dataset):
		continuous = dataset.getContinuousFeatures()
		def _translate(node,depth=0):
			if node.is_leaf:
				node.meaning = "Feature <%02d> Value <%02d>"%(target,node.name)
				try:
					node.meaning = dataset.getFeatureNumValueMeaning(target,node.name)
				except IndexError as e:
					LOGGER.warning("Unable to translate leaf with feature and value %d,%d",target,node.name)
				node.name = node.meaning
			else:
				# translate my children
				for child in node.children:
					_translate(child,depth+1)
				# translate me
				if depth % 2:
					# odd: is a feature value name
					if continuous[node.parent.name]:
						node.meaning = ">=%f"%node.parent.threshold if node.name else "<%f"%node.parent.threshold
					else:
						node.meaning = dataset.getFeatureNumValueMeaning(node.parent.name, node.name)
				else:
					# even: is a feature name
					node.meaning = dataset.getFeatureMeaning(node.name)
				node.name = node.meaning
		return _translate(tree)

	"""
	Given a tree it prunes the leaf nodes if necessary

	@param  trainingSet Set of samples to build the tree
	@param  tree		The tree to prune
	@param  target	  feature to be determined by the tree

	@return pr_tree	 the pruned tree if so
	"""
	def _treePruning (trainingSet, tree, target):
		pass

	"""
	Determines if the tree must stop growing and put a leaf or not, based on
	the current training set and attribute set remaining

	@return 	True if stop and put a leaf
	"""
	@abstractmethod
	def _stopCriterion(self, trainingSet, featureSet):
		pass

	"""
	Selects the next feature to classify the tree and returns a node for the tree containing that feature

	@param  trainingSet Set of samples to build the tree
	@return node   		tree node containing the feature to classify and its properties
	"""
	@abstractmethod
	def _splitCriterion(self, trainingSet, featureSet):
		pass

	"""
	Computes the entropy of the node, its a disorder indicator in order to be
	used as a split criterion, to select the next feature to be classified.

	@returns	h   the entropy value for the current assigment
	"""
	def _H(self, trainingSet):
		pass

"""
Dummy implementation of the tree growing algorithm. Generates a decision tree
with the simple criteria of picking the first feature as the next classifier
"""
class BasicTreeGrowingAlgorithm(TreeGrowingAlgorithm):
	def _stopCriterion(self, trainingSet, featureSet):
		# Not attributes remaining
		if not len(featureSet):
			return True
		# All elements belonging to a unique class
		if np.count_nonzero(self._countTargetClasses(trainingSet)) == 1:
			return True

		#Continue classifying
		return False

	def _splitCriterion(self, trainingSet, featureSet):
		feature = featureSet[0]
		featureNode = Node(feature)
		featureNode.featureValues = self._features[feature]
		featureNode.featureData = self._data[:,feature]
		featureNode.isContinuous = self._continuous[feature]
		featureNode.threshold = None
		return featureNode
