import pandas as pd
import numpy as np
import os

from src.utils.debug import Log, log, chalk, Status
from src.utils.series import Series

loadSeriesCache = {}

_log = Log()

class __Filer:
	def __init__(self):
		self.location = ""

	def setLocation(self, location: str):
		self.location = location

	def resolveFilePath(self, fileName: str, extension: str):
		try:
			dirPath = os.path.join(".","src",self.location, "results")
			os.mkdir(dirPath)
		except:
			pass
		return os.path.join(".","src",self.location, "results", f"{fileName}.{extension}")

	def writeTXT(self, fileName: str, writer):
		file = open(self.resolveFilePath(fileName,"txt"), "w")
		writer(file)
		file.close()

	def savefig(self, plot, fileName: str):
		plot.savefig(self.resolveFilePath(fileName,"png"))

	def loadCSV(self, filePath: str, **kwargs):
		return pd.read_csv(os.path.join(".", "data", filePath), sep=",", **kwargs)

	def loadSeries(self, filePath: str, **kwargs):
		if not filePath in loadSeriesCache:
			base = os.path.basename(filePath)
			(fileName, csv) = os.path.splitext(base)
			# print(f"{chalk.magenta(base)} - carregando arquivo")
			content = np.array([ val[1] for val in loadCSV(filePath, **kwargs).values] )

			loadSeriesCache[filePath] = (content, fileName)
			# print(f"{chalk.magenta(base)} - arquivo carregado")

		(content, name) = loadSeriesCache[filePath]
		return Series(values = content, name = name)

Filer = __Filer()