class Dataset(object):
	"""
	@attr 	_data   		data containing examples to train / validate
	@attr 	_classes		list of possible attributes of each column
	@attr 	_classes_names 	list of classes names
	@attr 	_classes_cnt	count of appearances of the attributes per class
							format: [class1,class2,..]
								class1: [attr1_count,attr2_count,...]
	@attr 	_isParsed 		true if already parsed
	"""
	__slots__ = ["_data","_classes","_classes_names","_classes_cnt","_isParsed"]

	"""
	Initializes a dataset given the matrix of examples

	@param 	data 	data containing examples to train / validate
	"""
	def __init__(self,data):
		self._data = data
		self._isParsed = False
		self._classes_names=[]

	"""
	Parses the dataset to find the possible classes and their attributes
	automatically
	"""
	def parse(self):
		# Create classes and attributes
		self._classes = [list(set([row[j] for row in self._data]))
			for j in range(len(self._data[0]))]
		# Count attributes
		self._isParsed = True

	def parseAttributes():
		translation = {}
		for element in data:
			parsed = element.split[':']
			class_name = parsed[0].strip()
			self._classes_names.append(class_name)
			values = parsed[1].strip()
			translation[class_name] = {}

			for el in values.split[',']
				long_name, short_name = el.split('=')
				translation[class_name][short_name.strip()] = long_name.strip()

	def __str__(self):
		txt =  " DATASET Specifications\n"
		txt += "------------------------------------------------------------\n"
		txt += "STAT:    "+("parsed" if self._isParsed else "init")+"\n"
		txt += "ROWS:    "+str(len(self._data))+"\n"
		txt += "COLS:    "+str(len(self._data[0]))+"\n"
		txt += "HEAD:    "+str(self._data[0])+"\n"
		txt += "TAIL:    "+str(self._data[-1])+"\n"
		return txt

	"""
	Returns true if has been parsed

	@return true / false depending on if has been parsed
	"""
	def isParsed(self):
		return self._isParsed
