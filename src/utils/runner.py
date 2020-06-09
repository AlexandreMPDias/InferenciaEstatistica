import pandas as pd
import numpy as np
import os
import time

from src.utils.debug import Log, log, chalk, Status

_log = log

statuses = ["ok", "skipped", "error"]

class Status:
	success = "ok"
	skip = "skipped"
	error = "error"

	options = ["ok", "skipped", "error"]

	@staticmethod
	def map(status):
		if status == "ok":
			return "sucessfully"
		if status == "skipped":
			return "not done"
		if status == "error":
			return "with error"

def runFunction(func, log = None, catch = True, timer = None, beforeEach = None, afterEach = None):
	timer = False if timer is None else timer
	funcName = func.__name__
	t0 = time.clock()

	xlog = _log if log is None else log

	if beforeEach is not None:
		beforeEach()

	try:
		xlog.print(f"{chalk.yellow(funcName)} - " + chalk.cyan("initiating", False))
		out = func()
		after = f"{chalk.yellow(funcName)} - " + chalk.lightgreen("finished", False)
		if out:
			after = f"{chalk.yellow(funcName)} - " + chalk.yellow("not done :c", False)
			xlog.print(after)
			return Status.skip
		if timer:
			elapsed = chalk.magenta("{:.2f}s".format(time.clock() - t0), False)
			after = f"{after} [ took: {elapsed} ]"
		xlog.print(after)
		if afterEach is not None:
			afterEach(True)
		return Status.success
	except Exception as err:
		if not catch:
			raise err
		xlog.print(f"{chalk.yellow(funcName)} - " + chalk.lightred("error", False))
		xlog.print(err)
		xlog.print("\n\n")
		if afterEach is not None:
			afterEach(False)
		return Status.error


def runFunctions(funcs: list, **kwargs):
	total = dict(skipped = 0, ok = 0, error = 0)
	for function in funcs:
		key = runFunction(function, **kwargs)
		total[key] = total[key] + 1
	return total


def questionRunner(questionsToRun, debug = False):
	total = dict(skipped = 0, ok = 0, error = 0)
	runs = 0

	def runSingle(question):
		qName = chalk.cyan(question.name)

		def message(m):
			print(f"{qName} - {m}")

		message("running")
		output = question.main(debug = debug)

		if output[Status.error] > 0:
			print(qName + " - " + chalk.lightred(Status.error, False))
		else:
			print(qName + " - " + chalk.lightgreen(Status.success, False))

		return output

	for question in questionsToRun:
		output = runSingle(question)
		for key in statuses:
			total[key] = total[key] + output[key]
			runs = runs + output[key]
		print("\n")

	print(f"Items ran: {runs}/{16}")
	colors = [chalk.lightgreen, chalk.yellow, chalk.lightred]
	for paint, key in zip(colors, statuses):
		count = total[key]
		if count > 0:
			colored = f"{Status.map(key)}: [ {count} / {runs} ]"
			print(f"Items finished {paint(colored)}")
