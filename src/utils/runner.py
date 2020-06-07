import pandas as pd
import numpy as np
import os
import time

from src.utils.debug import Log, log, chalk, Status

_log = log

def runFunction(func, log = None, catch = True, timer = None):
	timer = False if timer is None else timer
	funcName = func.__name__
	t0 = time.clock()

	xlog = _log if log is None else log

	try:
		xlog.print(f"{chalk.yellow(funcName)} - " + chalk.lightgreen("iniciando", False))
		func()
		after = f"{chalk.yellow(funcName)} - " + chalk.lightgreen("finalizada", False)
		if timer:
			elapsed = chalk.magenta("{:.2f}s".format(time.clock() - t0), False)
			after = f"{after} [ took: {elapsed} ]"
		xlog.print(after)
		return 0
	except Exception as err:
		if not catch:
			raise err
		xlog.print(f"{chalk.yellow(funcName)} - " + chalk.lightred("error", False))
		xlog.print(err)
		xlog.print("\n\n")
		return 1


def runFunctions(funcs: list, **kwargs):
	exitCode = 0
	for function in funcs:
		exitCode = exitCode + runFunction(function, **kwargs)
	return 1 if exitCode > 0 else 0
