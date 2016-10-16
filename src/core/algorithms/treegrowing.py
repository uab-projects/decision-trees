from abc import ABCMeta,abstractmethod,abstractproperty

class TreeGrowingAlgorithm(object):
    __metaclass__ = ABCMeta
    """
    """
    __slots__=[]

    """
    """
    def __init__(self):
        pass

    """
    Creates a decision tree built from a given training set to evaluate and
    classify some values in the future to determine the target value

    @param  trainingSet     list of samples already classified to build the tree
    @param  featureSet      list of features to evaluate
    @param  target          feature to be determined by the tree

    @return tree
    """
    def _treeGrowing(self, trainingSet, featureSet, target):

        # should be something diferent, specified like that on notes
        T = Node('root')

        if self._stopCriterion(trainingSet):
            pass
        else:
            next_attribute = self._splitCriterion(trainingSet)
            # temporary, just an idea -- born to help, not to work
            for possible_value in next_attribute:
                new_trainingSet = [ts for ts in trainingSet if ts[next_attribute] == possible_value]

                # Recursive call
                sub_tree = self._treeGrowing(new_trainingSet, featureSet, target)
                sub_tree = Node(possible_value, parent=T)

        return self._treePruning(trainingSet, T, target)


    """
    Given a tree it prunes the leaf nodes if necessary

    @param  trainingSet Set of samples to build the tree
    @param  tree        The tree to prune
    @param  target      feature to be determined by the tree

    @return pr_tree     the pruned tree if so
    """
    @abstractmethod
    def _treePruning ( trainingSet, tree, target):
        pass

    """
    <Some description in here>

    @return True if something, False if not something
    """
    @abstractmethod
    def _stopCriterion(self, trainingSet):
        pass

    """
    Selects the next attribute in order to classify the samples to determine the
    target value

    @param  trainingSet Set of samples to build the tree

    @return attribute   Next attribute to be classified
    """
    @abstractmethod
    def _splitCriterion(self, trainingSet):
        pass

    """
    Computes the entropy of the node, its a disorder indicator in order to be
    used as a split criterion, to select the next attribute to be classified.

    @returns    h   the entropy value for the current assigment
    """
    @abstractmethod
    def _entropy(self, trainingSet):
        pass
