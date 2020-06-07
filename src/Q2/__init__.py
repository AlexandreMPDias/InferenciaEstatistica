from os import path
from matplotlib import pyplot as plt
from probscale import plot_pos, probplot
import numpy as np
import math

from src.utils import chalk, loadSeries, loadCSV, runFunctions, Log, Filer, Series
from src.utils import constants
from src.Q1 import solveQuestion1, Gamma
from src.utils.estimators import Estimator

name = "Questão 2"
log = Log().instance(prepend = chalk.cyan("Q2"))
curr_location = path.join('.','src', 'Q2')

A1 = loadSeries("aluno3_A1.csv")

def probPlots(verbose = False):
	from scipy import stats

	for method in constants.EstimatorMethod.options:
		estimators = solveQuestion1(method)

		for estimator in estimators:
			# Create a new figure
			plt.figure()
			xlog = log.instance().indentMore()

			if(verbose):
				xlog.print("Plotting:").indentMore()
				xlog.print(f"density = {chalk.blue(estimator.density)}")
				xlog.print(f"method = {chalk.blue(estimator.method)}")
				xlog.print(f"parameter = {chalk.blue(estimator.parameter)}")
				xlog.indentLess()
			
			stats.probplot(
				x = A1.values,
				sparams = estimator.getDistributionParams(A1.values),
				dist = estimator.scipyDensity,
				plot = plt
			)

			name = f"[ProbabilityPlot] {estimator.density} - {estimator.method}"
			plt.title(name)

			# Set transparent dots in probability plot
			ax = plt.gca()
			line0 = ax.get_lines()[0]
			line0.set_alpha(0.25)

			Filer.savefig(plt, name)
			if verbose:
				xlog.success(f"finished - {name}")

def fill(size, content):
	return [content for i in range(0, size)]


def chiSquaredTest(serie: Series, verbose = False):
	from scipy import stats
	nChunks = [1,2,3,5,7,8,10]
	results = dict()
	for method in constants.EstimatorMethod.options:
		results[method] = dict()
		estimators = solveQuestion1(method)
		for estimator in estimators:
			results[method][estimator.density] = []

			for ddof in nChunks:

				worst_s, worst_p = None, None

				for chunk in np.array_split(serie.values, int(serie.length / 10)):
					values = serie.values

					values, chunk = chunk, values

					statistic, pvalue = stats.chisquare(
						values,
						fill(len(values), estimator.getEstimative(chunk)),
						ddof = ddof
					)

					if not math.isnan(pvalue):
						if worst_p is None:
							worst_p = pvalue
							worst_s = statistic
						elif pvalue > worst_p:
							worst_p = pvalue
							worst_s = statistic
				if worst_p is not None:
					results[method][estimator.density].append((ddof, worst_s, worst_p))

	return results

			


def q2_a():
	probPlots(True)

def q2_b():
	results = chiSquaredTest(serie = A1, verbose = True)

	def fileWrite(file):
		file.write("Q2: Teste Qui-quadrado\n")
		file.write("-=" * 30 + "-\n")
		for (method, m) in results.items():
			file.write(f"Metodo: [{method}]\n")
			for (density, d) in m.items():
				file.write(f"\tDensidade: [{density}]\n")
				file.write(f"\tGrau de Liberdade\tEstatistica\t Pvalue\n")
				for ((ddof, statistic, pvalue)) in d:
					s = "{:.4f}".format(statistic)
					p = format(pvalue, ".3e")
					file.write("\t{:^17}\t{:^11}\t{:^9}\n".format(str(ddof), s, p))
					# file.write(f"\t\tGrau de Liberdade: {ddof}\n")
					# file.write("\t\t\t- Estatistica de teste cumulativa de Pearson: {:.4f}\n".format(statistic))
					# file.write("\t\t\t- Pvalue: {}\n".format(format(pvalue, ".3e")))

	Filer.writeTXT("ChiSquareTest", fileWrite)

def main():
	Filer.setLocation("Q2")
	exitCode = runFunctions([
		# q2_a,
		q2_b
	], log = log, catch = False)
	return exitCode > 0