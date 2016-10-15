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
	__slots__=["_filename","_attr"]

	def __init__(self,filename):
		self._filename = filename

	def read(self):
		self._readFile()

	def _readFile(self):
		csvFile = open(self._filename, 'r')
		csvData = csv.reader(csvFile, delimiter=DATASET_SEP)
		self._data = list(csvData)

	def getAttr(self):
		return self._attr
