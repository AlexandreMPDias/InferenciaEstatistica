import pandas as pd
import numpy as np
import os

from src.utils.debug import Log, log, chalk, Status

def loadCSV(filePath: str, **kwargs):
	return pd.read_csv(os.path.join(".", "data", filePath), sep=",", **kwargs)

def loadSeries(filePath: str, **kwargs):
	return np.array([ val[1] for val in loadCSV(filePath, **kwargs).values] )

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
