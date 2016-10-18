class Dataset(object):
	"""
	@attr 	_data   		data containing examples to train / validate
	@attr 	_cols 			rows in the dataset	(number of items)
	@attr 	_cols			cols in the dataset (number of attributes)
	@attr 	_classes		list of possible attributes of each column
	@attr 	_attrSet	 	list of attributes and classes names
	@attr 	_classes_cnt	count of appearances of the attributes per class
							format: [class1,class2,..]
								class1: [attr1_count,attr2_count,...]
	@attr 	_isParsed 		true if already parsed
	"""
	__slots__ = ["_data","_rows","_cols","_classes","_attrSet",
	"_classes_cnt","_isParsed"]

	"""
	Initializes a dataset given the matrix of examples

	@param 	data 	data containing examples to train / validate
	"""
	def __init__(self,data):
		self._data = data
		self._rows = len(self._data)
		self._cols = len(self._data[0])
		self._isParsed = False
		self._attrSet = None
		self._parse()

	"""
	Parses the dataset to find the possible classes and their attributes
	automatically
	"""
	def _parse(self):
		# Create classes and attributes
		self._classes = [list(set([row[j] for row in self._data]))
			for j in range(len(self._data[0]))]
		# Count attributes
		self._isParsed = True

	"""
	Applies class names from an attributes set to give meaning to the attribute
	and its classes

	@param 	attrSet 	attribute set with human interesting info
	"""
	def applyAttributes(self,attrSet):
		self._attrSet = attrSet

	def __str__(self):
		txt =  " DATASET Specifications\n"
		txt += "------------------------------------------------------------\n"
		txt += "STAT:    %s\n"%("parsed" if self._isParsed else "init")
		txt += "SIZE:    %d x %d\n"%(self._rows,self._cols)
		txt += "HEAD:    %s\n"%(self._data[0])
		txt += "TAIL:    %s\n"%(self._data[-1])
		txt += "CLASSES: %d\n"%(len(self._classes))
		i = 1
		for class_attr in self._classes:
			txt += " [%02d]: %s\n"%(i,class_attr)
			i+=1
		return txt

	"""
	Returns true if has been parsed

	@return true / false depending on if has been parsed
	"""
	def isParsed(self):
		return self._isParsed

	"""
	Returns the number of rows in the dataset

	@return 	number of rows
	"""
	def getRows(self):
		return self._rows

	"""
	Returns the number of cols in the dataset

	@return 	number of cols
	"""
	def getCols(self):
		return self._cols

	def getData(self):
		return self._data

	def getClasses(self):
		return self._classes

	def getAttributeSet(self):
		return self._attrSet
