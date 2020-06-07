import pandas as pd
import numpy as np
import os
import time


from src.utils.debug import Log, log, chalk, Status
from src.utils.series import Series
from src.utils.filer import Filer
from src.utils.runner import runFunction, runFunctions

loadSeriesCache = {}


def loadCSV(filePath: str, **kwargs):
	return pd.read_csv(os.path.join(".", "data", filePath), sep=",", **kwargs)

def loadSeries(filePath: str, **kwargs):
	if not filePath in loadSeriesCache:
		base = os.path.basename(filePath)
		(fileName, csv) = os.path.splitext(base)
		# print(f"{chalk.magenta(base)} - carregando arquivo")
		content = np.array([ val[1] for val in loadCSV(filePath, **kwargs).values] )

		loadSeriesCache[filePath] = (content, fileName)
		# print(f"{chalk.magenta(base)} - arquivo carregado")

	(content, name) = loadSeriesCache[filePath]
	return Series(values = content, name = name)
