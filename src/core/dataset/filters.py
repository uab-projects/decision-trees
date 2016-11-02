# libraries
import logging

# constants
LOGGER = logging.getLogger(__name__)

"""
Ables to filter a dataset in order to clean the dataset data from non-valid or non-desirable feature values
"""
class DatasetFilterer(object):
	"""
	@attr   _data		the dataset data to filter
	"""
	__slots__ = ["_data"]

	"""
	Initializes the filterer with the data specified

	@param  data	data to filter
	"""
	def __init__(self, data):
		self._data = data

	"""
	Removes the rows with unknown values

	@param  unk		the character to use when there's an unknown value
	"""
	def deleteUnknownSamples(self, unk='?'):
		LOGGER.debug("Before removing %d",len(self._data))
		# remove rows with unk char
		self._data = list(filter(lambda s: unk not in s, self._data))
		LOGGER.debug("After removing: %d",len(self._data))

	"""
	Returns the data (filtered or not)

	@return 	data
	"""
	def getData(self):
		return self._data
