import math
import numpy
from src.utils import chalk, loadSeries, loadCSV, runFunctions
from src.utils import constants
from src.utils.estimators import Matcher, Estimation

class Combined:
	def __init__(self, density, method, parameter):
		self.density = density
		self.method = method
		self.parameter = parameter

		self.estimation = Estimation(density=self.density, method = self.method)
		self.estimator = self.estimation.getEstimator(self.parameter)

	def setMethod(self, method):
		self.method = method
		self.estimation = Estimation(density=self.density, method = self.method)
		self.setParameter(self.parameter)
		return self

	def setParameter(self, parameter):
		self.parameter = parameter
		self.estimator = self.estimation.getEstimator(self.parameter)
		return self


class Gamma(Combined):
	def __init__(self):
		super().__init__(
			density = constants.Density.gamma,
			method = constants.EstimatorMethod.likelihood,
			parameter = constants.Parameter.theta
		)

class Exponential(Combined):
	def __init__(self):
		super().__init__(
			density = constants.Density.exponential,
			method = constants.EstimatorMethod.likelihood,
			parameter = constants.Parameter.lamb
		)

class LogNormal(Combined):
	def __init__(self):
		super().__init__(
			density = constants.Density.logNormal,
			method = constants.EstimatorMethod.likelihood,
			parameter = constants.Parameter.sigma
		)	

def solveQuestion1(method):
	combineds = [ Gamma(), Exponential(), LogNormal() ]
	for combined in combineds:
		combined.setMethod(method)
	return [ combined.estimator for combined in combineds]

def assertTest(name, expected, received):
	for prop in received:
		rec = received[prop]
		exp = expected[prop]
		if not rec == exp:
			print("expected", chalk.green(str(expected)))
			print("\n")
			print("received", chalk.green(str(received)))
			raise Exception("Questão 1: teste_gamma failed")

class Q1Tests:
	def __init__(self, debug):
		self.debug = debug

	def test_gamma(self):
		if not self.debug:
			return
		gamma = Gamma()
		for method in constants.EstimatorMethod.options:
			est = gamma.setMethod(method).estimator
			received = dict(
				density = est.density,
				parameter = est.parameter,
				method = est.method
			)

			expected = dict(
				density = constants.Density.gamma,
				parameter = constants.Parameter.theta,
				method = method
			)
			assertTest("gamma", expected = expected, received = received)

	def test_exp(self):
		if not self.debug:
			return
		exponential = Exponential()
		for method in constants.EstimatorMethod.options:
			est = exponential.setMethod(method).estimator
			received = dict(
				density = est.density,
				parameter = est.parameter,
				method = est.method
			)

			expected = dict(
				density = constants.Density.exponential,
				parameter = constants.Parameter.lamb,
				method = method
			)
			assertTest("exponential", expected = expected, received = received)

	def test_logN(self):
		if not self.debug:
			return
		logNormal = LogNormal()
		for method in constants.EstimatorMethod.options:
			est = logNormal.setMethod(method).estimator
			received = dict(
				density = est.density,
				parameter = est.parameter,
				method = est.method
			)

			expected = dict(
				density = constants.Density.logNormal,
				parameter = constants.Parameter.sigma,
				method = method
			)
			assertTest("logNormal", expected = expected, received = received)

name = "Questão 1"
def main(debug):
	tests = Q1Tests(debug)
	return runFunctions([
		tests.test_gamma,
		tests.test_exp,
		tests.test_logN
	])