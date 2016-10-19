import math

#Basic ID3 algorithm version
class ID3(BasicTreeGrowingAlgorithm):
	def _splitCriterion(self, trainingSet, attributeSet):
        #create count-list for every attribute remaining in attributeSet

        #check for all the entropies
        for attr in attributeSet:
            #calculate entropy for the next assigment

        #return the attribute with the maximum gain (minimum entropy) if any
		return attributeSet[0]

    """
    Calculates and returns the entropy of the given values taken as a list

    @param  amounts     list of the amounts of elements for one attribute
                        belonging the target class
    """
    def _H(amounts):
        acc=0
        total_classes = sum(amounts)

        for el in amounts:
            prob = el/total_classes
            acc -= -prob * math.log(prob,2)
        return acc

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
            return
