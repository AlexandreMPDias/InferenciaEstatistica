
import math
import numpy
from src.utils.constants import Density, EstimatorMethod, Parameter

	
class Estimator:

	def __init__(self,density,parameter,method,formula):
		self.density = density
		self.parameter = parameter
		self.method = method
		self.formula = formula
		self.scipyDensity = Matcher(density = density).getScipyDensity()

	def getEstimative(self,randomSample):
		return self.formula(randomSample)

	def loadDistributionParams(self, values):
		if (self.density == Density.gamma):
			a = 2
			loc = 0
			scale = 1.0/self.getEstimative(values)
			self.distributionParams = dict(
				a = a,
				loc = loc,
				scale = scale,
			)

		elif (self.density == Density.exponential):
			loc = 0
			scale = 1.0/self.getEstimative(values)
			self.distributionParams = dict(
				loc = loc,
				scale = scale,
			)

		elif (self.density == Density.logNormal):
			s = self.getEstimative(values)
			loc = 0
			scale = 1
			self.distributionParams = dict(
				s = s,
				loc = loc,
				scale = scale,
			)

	def getDistributionParams(self, values):
		self.loadDistributionParams(values)
		if (self.density == Density.gamma):
			a = self.distributionParams[self.parameter]
			loc = self.distributionParams['loc']
			scale = self.distributionParams['scale']
			return (a, scale)

		elif (self.density == Density.exponential):
			loc = self.distributionParams['loc']
			scale = self.distributionParams['scale']
			return (scale)

		elif (self.density == Density.logNormal):
			s = self.distributionParams['s']
			loc = self.distributionParams['loc']
			scale = self.distributionParams['scale']
			return (s, loc,scale)

	def getCDF(self, values): 
		from scipy import stats

		self.loadDistributionParams(values)
		locScale = dict(
			loc = self.distributionParams['loc'],
			scale = self.distributionParams['scale']
		)

		if (self.density == Density.gamma):
			def cdf(x):
				return stats.gamma.cdf(x, 2, **locScale)

		elif (self.density == Density.exponential):
			def cdf(x):
				return stats.expon.cdf(x, **locScale)

		elif (self.density == Density.logNormal):
			sigma = self.getEstimative(values)
			def cdf(x):
				return stats.lognorm.cdf(x, sigma, **locScale)
		return cdf

	def getPPF(self, values): 
		from scipy import stats

		self.loadDistributionParams(values)
		locScale = dict(
			loc = self.distributionParams['loc'],
			scale = 1/self.distributionParams['scale']
		)

		if (self.density == Density.gamma):
			def ppf(x):
				return stats.gamma.ppf(x, 2, **locScale)

		elif (self.density == Density.exponential):
			locScale['scale'] = self.distributionParams['scale']
			def ppf(x):
				return stats.expon.ppf(x, **locScale)

		elif (self.density == Density.logNormal):
			sigma = self.getEstimative(values)
			def ppf(x):
				return stats.lognorm.ppf(x, sigma, **locScale)
		return ppf

	def __str__(self):
		values = dict(
			density = self.density,
			parameter = self.parameter,
			method = self.method
		)
		strg = []
		for (key, value) in values.items():
			strg.append(f"\t{key} = {value}")
		return "{\n" + "\n".join(strg) + "\n}"


class Estimation:

	def __init__(self, density = None, method = None):
		self.density = density
		self.method = method

	def getEstimator(self, parameter):
		self.parameter = parameter

		matcher = Matcher(parameter = parameter, density = self.density, method = self.method)

		if(matcher.match(density = Density.logNormal)):
			if(matcher.match(method = EstimatorMethod.likelihood)):

				def estimatorFormula(randomSample):
					randomSampleLog = numpy.log(randomSample)
					return numpy.mean(randomSampleLog ** 2)
			else:
				def estimatorFormula(randomSample):
					return numpy.sqrt(2*numpy.log(numpy.mean(randomSample)))
				
		elif(matcher.match( density=Density.gamma )):

			def estimatorFormula(randomSample):
				return 0.5*numpy.mean(randomSample)
		elif(matcher.match( density = Density.exponential )):

			def estimatorFormula(randomSample):
				return 1.0/numpy.mean(randomSample)
		else:
			raise Exception("Nothing was matched for: " + str(dict(
				parameter = parameter,
				density = self.density,
				method = self.method
			)))

		estimator = Estimator(density = self.density, parameter = parameter, method = self.method, formula = estimatorFormula)
		return estimator


class Matcher:
	def __init__(self, **estimation):
		self.estimation = estimation

	def isDensity(self, *options):
		out = False
		for option in options:
			if option not in Density.options:
				raise Exception(f"Invalid Density {option}")
			if option == self.estimation['density']:
				out = True
		return out

	def isParameter(self, *options):
		out = False
		for option in options:
			if option not in Parameter.options:
				raise Exception(f"Invalid Parameter {option} for density {self.estimation['density']}\nValid Parameters: [{','.join(Parameter.options)}]")
			try:
				if option == self.estimation['parameter']:
					out = True
			except:
				out = False
		return out

	def isMethod(self, *options):
		out = False
		for option in options:
			if option not in EstimatorMethod.options:
				raise Exception(f"Invalid EstimatorMethod {option}")
			if option == self.estimation['method']:
				out = True
		return out

	def match(self, density = None, method = None):
		out = True
		accepted = Density.parametersOf(self.estimation['density'])
		matchers = dict(
			density = (self.isDensity, density),
			parameter = (self.isParameter, accepted),
			method = (self.isMethod, method)
		)
		for (matcher, options) in matchers.values():
			if options is not None:
				opts = [options] if isinstance(options, str) else options
				out = out and matcher(*opts)
		return out

	def getScipyDensity(self):
		from scipy import stats
		density = self.estimation['density']

		if (density == Density.gamma):
			return stats.gamma
		
		elif (density == Density.exponential):

			return stats.expon
		
		elif (density == Density.logNormal):

			return stats.lognorm

