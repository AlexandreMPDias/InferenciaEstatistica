import math

class Series:
	def __init__(self, values = [], name = ""):
		self.name = name
		self.values = values
		self.max = max(*values)
		self.min = min(*values)
		self.length = len(values)
		self.total = 0
		for value in values:
			self.total = self.total + value
		self.avg = self.total / self.length

		self.var = self.__getVariance()

		self.std_deviation = math.sqrt(self.var)


	def apply(self, funcao):
		values = [funcao(x) for x in self.values]
		return Series(
			values = values,
			name = self.name
		)

	def __getVariance(self):
		diffs = [math.pow(x - self.avg ,2) for x in self.values]
		n = self.length - 1
		return sum(diffs)/ n
