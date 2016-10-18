import csv
from .constants import *

class DataReader(object):

	__slots__=["_filename","_data"]

	def __init__(self,filename):
		self._filename = filename

	def read(self):
		self._readFile()

	def _readFile(self):
		csvFile = open(self._filename, 'r')
		csvData = csv.reader(csvFile, delimiter=DATASET_SEP)
		self._data = list(csvData)

	def getData(self):
		return self._data

class AttrReader(object):
	__slots__=["_filename","_data","_attr","_isParsed","_isRead"]

	def __init__(self,filename):
		self._filename = filename
		self._data = None
		self._attr = []
		self._isRead = False
		self._isParsed = False

	def read(self):
		self._readFile()
		self._isRead = True

	def parse(self):
		assert self._isRead, "unable to parse if file has not been read"
		self._parseFile()
		self._isParsed = True

	def _readFile(self):
		self._data = [line.rstrip('\n') for line in open(self._filename,"r")]

	"""
	Reads from the file given the attributes and loads them into the class to
	after that be able to show more informative and human data
	"""
	def _parseFile(self):
		self._attr = []
		for class_def in self._data:
			attr_name, class_values = class_def.split(':')
			attr = [attr_name.strip(),{}]
			for class_names in class_values.split(','):
				long_name, short_name = class_names.split('=')
				attr[1][short_name.strip()] = long_name.strip()
			self._attr.append(attr)

	def getAttr(self):
		return self._attr

	def isParsed(self):
		return self._isParsed
