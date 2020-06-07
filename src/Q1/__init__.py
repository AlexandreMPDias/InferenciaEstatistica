import math
import numpy
from src.utils import chalk, loadSeries, loadCSV, runFunctions
from src.utils import constants
from src.utils.estimators import Matcher, Estimation

	
# class Estimator:

# 	def __init__(self,density,parameter,method,formula):
# 		self.density = density
# 		self.parameter = parameter
# 		self.method = method
# 		self.formula = formula
# 		self.scipyDensity = Matcher(density = density).getScipyDensity()

# 	def getEstimative(self,randomSample):

# 		return self.formula(randomSample)

# 	def __str__(self):
# 		values = dict(
# 			density = self.density,
# 			parameter = self.parameter,
# 			method = self.method
# 		)
# 		strg = []
# 		for (key, value) in values.items():
# 			strg.append(f"\t{key} = {value}")
# 		return "{\n" + "\n".join(strg) + "\n}"


# class Estimation:

# 	def __init__(self, density = None, method = None):
# 		self.density = density
# 		self.method = method

# 	def getEstimator(self, parameter):
# 		self.parameter = parameter

# 		matcher = Matcher(parameter = parameter, density = self.density, method = self.method)

# 		if(matcher.match(
# 			density = constants.Density.logNormal,
# 		)):
# 			if(matcher.match(method = constants.EstimatorMethod.likelihood)):

# 				def estimatorFormula(randomSample):
# 					randomSampleLog = numpy.log(randomSample)
# 					return numpy.mean(randomSampleLog ** 2)
# 			else:

# 				def estimatorFormula(randomSample):
# 					return numpy.sqrt(2*numpy.log(numpy.mean(randomSample)))
				
# 		elif(matcher.match( density=constants.Density.gamma )):

# 			def estimatorFormula(randomSample):
# 				return 0.5*numpy.mean(randomSample)
# 		elif(matcher.match( density = constants.Density.exponential )):

# 			def estimatorFormula(randomSample):
# 				return 1.0/numpy.mean(randomSample)
# 		else:
# 			raise Exception("Nothing was matched for: " + str(dict(
# 				parameter = parameter,
# 				density = self.density,
# 				method = self.method
# 			)))

# 		estimator = Estimator(density = self.density, parameter = parameter, method = self.method, formula = estimatorFormula)
# 		return estimator


# class SolveQ1:

# 	def __init__(self, method, parameter):
# 		self.gammaEstimator = Estimation(density=constants.Density.gamma,method=method).getEstimator(parameter = constants.Parameter.theta)
# 		self.exponentialEstimator = Estimation(density=constants.Density.exponential,method=method).getEstimator(parameter = constants.Parameter.lamb)
# 		self.logNormalEstimator = Estimation(density=constants.Density.logNormal,method=method).getEstimator(parameter = constants.Parameter.sigma)

# 	def values(self):
# 		return [self.gammaEstimator,self.exponentialEstimator,self.logNormalEstimator]
		
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

def test_gamma():
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

def test_exp():
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

def test_logN():
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
def main():
	return runFunctions([
		test_gamma,
		test_exp,
		test_logN
	])