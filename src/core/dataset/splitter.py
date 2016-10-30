# libraries
import random

LOGGER = logging.getLogger(__name__)

"""
Given a dataset, generates a training set and a validation set according to
the splitting method specified
"""
class DatasetSplitter(object):
	"""
	@attr   _dataset	original dataset
	"""
	__slots__ = ["_dataset"]

	"""
	Initializes a new dataset splitter with the dataset to split specified as
	a parameter
	"""
	def __init__(self, dataset):
		self._dataset = dataset

	"""
	Generates a training set and a validation set using the holdout method, this
	means, specifying a percent of the dataset, we will generate a random
	training set that contains that percentage of the dataset and the rest will
	be the validation set

	@param  percent	 percent of random samples to pick from the dataset to
						set as the training set
	@return trainingSet and validationSet objects generated
	"""
	def holdout(self, percent):
		# shuffle dataset data
		dataset = self._dataset.getData()
		np.random.shuffle(dataset)
		rows = dataset.getRows()
		tr = int(rows*percent)

		LOGGER.debug("training set has %d elements, validation set has %d elements from a total of %d " %(tr, rows-tr,rows))
		LOGGER.debug("first element of the training set is: ",dataset[0])
		LOGGER.debug("first element of the validation set is: ",dataset[tr])
		return (dataset[:tr], dataset[tr:])

	"""
	"""
	def crossValidation(self, k):
		pass


"""
IDEA
if args.validation_algorithm == "holdout":
	sets = dataset.getSets(trainingSet_pc)
	trainingSet = TrainingSet(sets[0])
	validationSet = ValidationSet(sets[1],args.classifier)

	tree = algorithm(args.classifier)
	accuracy = validationSet.validateTree(tree)
elif args.validation_algorithm == "cross-validation":
	k = args.k
	p = 1/k
	tr = k-1*p
	vd = p
	sets = dataset.getSets(1)

	full_dts = sets[0]
	size = len(full_dts)

	size_partition= size / k
	parts = []
	for i in range(k):
		parts.append(full_dts[i*size_partition:(i+1)*size_partition])

	accuracy = -1.0
	a = []
	for i in range(k):
		tr =parts[:i]+parts[i+1:]
		vd = parts[i]
		trainingSet = TrainingSet(tr)
		validationSet = ValidationSet(vd,args.classifier)
		t = algorithm(args.classifier)
	   	a.append(validationSet.validateTree(tree))

		if a[i]> accuracy:
			accuracy = a[i]
			tree = t

	tree t es el mejor con la mejor accuracy a
"""
