from src.utils.series import Series
from src.utils.estimators import Estimator
from src.utils.formatter import Format


def fill(size, content):
	return [content for i in range(0, size)]


class Interval:
	def __init__(self, index, lowerLimit = None, upperLimit = None):
		"""
			@param {int} index - the index of the interval
			@param {lowerLimit} - the value of the smallest possible value in the interval
			@param {upperLimit} - the value of the greatest possible value in the interval
			if upperLimit is not, upperLimit is set to [infinity]
		"""
		self.index = index
		self.lower = 0 if lowerLimit is None else lowerLimit
		self.upper = upperLimit

	def contains(self, value):
		return self.__isAhead(value) and self.__isBefore(value)

	def probabilityOfBeingInsideInterval(self, cdf):
		probGreaterThanSmallest = cdf(self.lower)
		probSmallerThanGreatest = 1
		if self.upper is not None:
			probSmallerThanGreatest = cdf(self.upper)
		return probSmallerThanGreatest - probGreaterThanSmallest

	def __isAhead(self, value):
		return value >= self.lower

	def __isBefore(self, value):
		if self.upper is None:
			return True
		return value < self.upper

	def __str__(self):
		l = str(self.lower)
		u = "Inf" if self.upper is None else str(self.upper)
		i = str(self.index)
		return "{ "+ i +" }: " + f"[ {l}.00 , {u}.00 )"

class Frequency:
	def __init__(self, serie: Series, ranges: list):
		self.serie = serie

		self.ranges = ranges
		self.__intervals = self.__createIntervalFromList(ranges)

		self.observed = self.__groupBy()
		self.expected = []

	def setEstimator(self, estimator: Estimator):
		frequencies = []

		cdf = estimator.getCDF(self.serie.values)
		if estimator.density == "Gamma":
			from scipy import stats
			d = estimator.distributionParams
			d['scale'] = 1/d['scale']
			def newCdf(x):
				return stats.gamma.cdf(x, **d)
			cdf = newCdf

		for interval in self.__intervals:
			prob = interval.probabilityOfBeingInsideInterval(cdf)
			frequencies.append(prob)

		self.expected = frequencies
		return frequencies
	
	def __groupBy(self):
		groups = fill(len(self.__intervals), 0)

		for value in self.serie.values:
			interval = self.__findInterval(value)
			groups[interval.index] = groups[interval.index] + 1
			
		return  [g/self.serie.length for g in groups]

	def __createIntervalFromList(self, ranges: list):
		intervals = []
		lastIndex = len(ranges)
		for i in range(0, lastIndex + 1):
			lower = None if i == 0 else ranges[i - 1]
			upper = None if i == lastIndex else ranges[i]
			intervals.append(Interval(i, lower, upper))
		return intervals

	def __findInterval(self, value):
		for interval in self.__intervals:
			if interval.contains(value):
				return interval

	def __str__(self, indentLevel = 1):
		out = []
		t = "\t" * indentLevel

		for interval in self.__intervals:
			i = str(interval)
			o = Format.percent(self.observed[interval.index])
			c = int(self.observed[interval.index] * self.serie.length)
			f = Format.percent(self.expected[interval.index])
			out.append(f"{t}{i} - [ {o} ({c}) :: {f} ]")
		return "\n".join(out)
			

			
		