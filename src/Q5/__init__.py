import math
from os import path
from src.utils import chalk, loadSeries, loadCSV, runFunctions, Log, Series, Filer
from src.utils.estimators import Estimator
from src.utils import constants
from src import Q1
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import scale as matplotlibScale
from probscale import plot_pos, probplot

name = "Questão 5"
log = Log().instance(prepend = chalk.cyan("Q5"))
curr_location = path.join('.','src', 'Q5')

A2 = loadSeries("aluno3_A2.csv")
A3 = loadSeries("aluno3_A3.csv")

name = "Questão 5"


class FisherSnedecor:
	def __init__(self, dfn, dfd = 1):
		"""
			Initialize the FisherSnedecor class

			@param {float} F_obs: Observer F's value
			@param {int} dfn: degrees of freedom of the numerator
			@param {int} dfd: degrees of freedom of the denominator

			> Properties:
				- {float} F_obs: Observer F's value
				- {float} F_tab: The table value of F
				- {int} dfn: degrees of freedom of the numerator
				- {int} dfd: degrees of freedom of the denominator
		"""
		self.dfn = dfn
		self.dfd = dfd

	def pdf(self, x):
		from scipy.stats import f
		return f.pdf(x, self.dfn, self.dfd)

	def ppf(self, q):
		from scipy.stats import f
		return f.ppf(q, self.dfn, self.dfd)
	
	def cdf(self,x):
		from scipy.stats import f
		return f.cdf(x, self.dfn, self.dfd)

class HypothesisTesting:
	def __init__(self,significance):
		self.significance = significance

	def testEqualVariance(self):
		from scipy.stats import  f

		F = A2.var / A3.var

		statisticsCDF = f.cdf(F, A2.length - 1, A3.length - 1)
		"""
			Accumulated value of the F distribution when the test's statistics are replaced.
			Essencially, it's the accumulated probability until the test statistic.
		"""		
		
		dontReject = (statisticsCDF > self.significance/2) and (statisticsCDF < 1 - self.significance/2)
		"""
			if:
				- statisticsCDF is bigger than (alpha/2)
				- statisticsCDF is smaller than (1 - alpha/2)

				then:
					dontReject the Ho hypothesis [s1/s2 = 1]
		"""
		return dontReject

	def typeIIErrorProb(self,varianceRatio,significance, f: FisherSnedecor):

		"""
			Calculates Type II Error Probability (Beta) given the ration between
			populational variances.

			@param {float} varianceRatio: Ratio between populational variances
			@param {int} significance: Statistical test's pre-defined significance
			(Type I Error Probability)
			@param {FisherSnedecor} f: FisherSnedecor distribution data and properties
		"""

		delimiter = f.ppf(q = 1 - significance/2)

		if (varianceRatio > 0 and varianceRatio < 1):

			beta = f.cdf(x = delimiter/varianceRatio) - f.cdf(x = delimiter)

		
		elif (varianceRatio >= 1 and varianceRatio < (delimiter ** 2)):

			beta = f.cdf(x = 1.0/delimiter) - f.cdf(x= 1.0/(varianceRatio*delimiter))
		

		elif(varianceRatio >= (delimiter ** 2)):

			beta = f.cdf(x = delimiter/varianceRatio) - f.cdf(x = 1.0/(varianceRatio*delimiter))

		return beta


	def plotCaracteristicCurve(self, f: FisherSnedecor):
		"""
			Plots test's operating-characteristic, i.e., curve where x-axis contains
			variance ratios and y-axis contains Type II Error Probabilities for given
			variance ratios.

			@param {FisherSnedecor} f: FisherSnedecor distribution data and properties
		"""


		fig, ax = plt.subplots(1,1)
		name = "Caracteristic Curve"
		
		varianceRatios = np.arange(start = 0.1, stop = 10, step = 0.1)

		betas = [self.typeIIErrorProb(varianceRatio,self.significance,f) for varianceRatio in varianceRatios]

		ax.plot(varianceRatios, betas, 'b-', lw=1, alpha=0.6)

		Filer.savefig(plt, name)
		plt.close()


	def getFisherDistribution(self):
		"""
			Get a FischerDistribution class with the values of the
			[degrees of freedom of the numerator] and [degrees of freedom of the denominator]
			already set
		"""
		n , m = A2.length, A3.length
		s_1, s_2 = A2.std_deviation, A3.std_deviation

		return FisherSnedecor(n-1, m-1)

	def confidenceInterval(self, f: FisherSnedecor):
		mean_1, mean_2 = A2.avg, A3.avg
		s_1, s_2 = A2.std_deviation, A3.std_deviation

		F_alpha_left = f.ppf(self.significance/2)

		F_alpha_right = f.ppf(1 - self.significance/2)

		IC = [f.F_obs/F_alpha_right, f.F_obs/F_alpha_left]

		return IC

h = HypothesisTesting(0.05)

def q5_a():
	dontReject = h.testEqualVariance()
	significance = "{:.2}".format(h.significance)

	xlog = log.instance()
	title = "(a) - Test Equal Variance"
	xlog.print(chalk.cyan(title)).indentMore()
	xlog.print(f"- For significance = {chalk.cyan(significance)}")
	if dontReject:
		xlog.print("[ Null Hypothesis was " + chalk.green("not rejected") + " ]")
	else:
		xlog.print("[ Null Hypothesis was " + chalk.red("rejected") + " ]")

	def writeFile(file):
		file.write(title + "\n")
		file.write(f"For significance = {significance}\n")
		out = "not " if dontReject else ""
		file.write(f"\tNull Hypothesis was {out}rejected")

	Filer.writeTXT("q5_a", writeFile)

def q5_b():
	title = "(b) - Type II Error"
	xlog = log.instance(prepend = chalk.cyan(title)).indentMore()

	distribution = h.getFisherDistribution()
	h.plotCaracteristicCurve(distribution)

	typeIIErrorProb = h.typeIIErrorProb(0.9 , h.significance, distribution)
	formattedProp = "{:.4}".format(typeIIErrorProb)

	def writeFile(file):
		file.write(title + "\n")
		base = "Probability of Type 2 Error = "
		file.write(f"{base}{formattedProp}")
		xlog.print(f"{base}{chalk.green(formattedProp)}")

	Filer.writeTXT("q5_b", writeFile)

def q5_c():
	pass

def main():
	Filer.setLocation("Q5")
	exitCode = runFunctions([
		q5_a,
		q5_b,
		q5_c
	], catch = False)
	return exitCode > 0