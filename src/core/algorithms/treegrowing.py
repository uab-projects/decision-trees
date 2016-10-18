from abc import ABCMeta,abstractmethod,abstractproperty
from anytree import *
import numpy as np
import random

class TreeGrowingAlgorithm(object):
	__metaclass__ = ABCMeta
	"""
	"""
	__slots__=["_target","_trainingSet","_trainingData","_trainingClasses","_isRunning"]

	"""
	"""
	def __init__(self, trainingSet):
		self._target = None
		self._trainingSet = trainingSet
		self._trainingClasses = trainingSet.getClasses()
		self._trainingData = trainingSet.getData()
		self._isRunning = False

	def __call__(self, target):
		# not running
		assert not self._isRunning, "the algorithm is alredy running"
		self._isRunning = True
		# generate attributes
		attributeSet = [i for i in range(self._trainingSet.getCols())]
		# check correct target
		assert target in attributeSet, """target attribute is not defined in
		attribute set, target must be %d <= target < %d"""%(0,len(attributeSet))
		self._target = target
		attributeSet.remove(target)
		# classify
		tree = self._treeGrowing(np.ones(self._trainingSet.getRows(),
			dtype=bool), attributeSet)
		self._isRunning = False
		return tree.children[0]

	"""
	Creates a decision tree built from a given training set to evaluate and
	classify some values in the future to determine the target value

	@param  trainingSet	 	list of boolean references of the training set
	@param  attributeSet 	list of features to evaluate
	@param  target		 	feature to be determined by the tree

	@return tree
	"""
	def _treeGrowing(self, trainingSet, attributeSet, parent=Node("root")):
		# should be something diferent, specified like that on notes
		if self._stopCriterion(trainingSet, attributeSet):
			# fulles
			Node("Solution %d"%(random.randrange(2)),parent)
		else:
			attr_index = self._splitCriterion(trainingSet, attributeSet)
			attributeSet.remove(attr_index)
			attr_node = Node(attr_index,parent)
			# temporary, just an idea -- born to help, not to work
			for attr_class in range(len(self._trainingClasses[attr_index])):
				# cut training set
				trainingSet *= np.where(self._trainingData[:,attr_index] == attr_class,True,False)
				# Recursive call
				self._treeGrowing(trainingSet, attributeSet, Node(attr_class,attr_node))
		return parent

	"""
	Applies attributes (if any) from the training set in order to transform a
	numerical weird meaning-less tree into something useful

	@param 	tree 	tree to translate
	"""
	def translate(self,tree):
		attrs = self._trainingSet.getAttributeSet()
		classes = self._trainingSet.getClasses()
		if not attrs:
			return None
		def _translate(node,depth=0):
			if not node.children:
				return
			else:
				# translate my children
				for child in node.children:
					_translate(child,depth+1)
				# translate me
				if depth % 2:
					# odd: is a class name
					short_name = classes[node.parent.name][node.name]
					node.name = attrs[node.parent.name][1][short_name]
				else:
					# even: is a attribute name
					node.name = attrs[node.name][0]
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
	<Some description in here>

	@return True if something, False if not something
	"""
	@abstractmethod
	def _stopCriterion(self, trainingSet, attributeSet):
		pass

	"""
	Selects the next attribute in order to classify the samples to determine the
	target value

	@param  trainingSet Set of samples to build the tree

	@return attribute   Next attribute to be classified
	"""
	@abstractmethod
	def _splitCriterion(self, trainingSet, attributeSet):
		pass

	"""
	Computes the entropy of the node, its a disorder indicator in order to be
	used as a split criterion, to select the next attribute to be classified.

	@returns	h   the entropy value for the current assigment
	"""
	def _H(self, trainingSet):
		pass

class BasicTreeGrowingAlgorithm(TreeGrowingAlgorithm):
	def _stopCriterion(self, trainingSet, attributeSet):
		if not len(attributeSet):
			return True
		else:
			pass
		return False
	def _splitCriterion(self, trainingSet, attributeSet):
		return attributeSet[0]
