import pandas as pd
import numpy as np
import os

from src.utils.debug import Log, log, chalk, Status
from src.utils.series import Series

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

def runFunction(func):
	funcName = func.__name__
	try:
		print(f"{chalk.yellow(funcName)} - " + chalk.lightgreen("iniciando", False))
		func()
		print(f"{chalk.yellow(funcName)} - " + chalk.lightgreen("finalizada", False))
		return 0
	except Exception as err:
		print(f"{chalk.yellow(funcName)} - " + chalk.lightred("error", False))
		print(err)
		print("\n\n")
		return 1


def runFunctions(funcs: list):
	exitCode = 0
	for function in funcs:
		exitCode = exitCode + runFunction(function)
	return 1 if exitCode > 0 else 0
