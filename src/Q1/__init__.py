import math
from src.utils import chalk, loadSeries, loadCSV, runFunctions

def sum_xi(X,n, x, transform = None):
	def fallback_transform(x):
		return x

	if len(X) < 1:
		raise Exception("Error: Vetor de Variaveis aleatórias possui dimensão 0.")
	t = fallback_transform if transform is None else transform
	return sum([t(X[i](x)) for i in range(0, n)])
	

class Estimador:
	def __init__(self, name = "", vero = None, momento = None):
		self.name = name
		self.vero = vero
		self.momento = momento

def getGamma():
	def vero(*X):
		from scipy.stats import gamma
		def inner(x):
			return gamma.cdf(x,2)
		return inner

	def momento(*X):
		def inner(x):
			n = len(X)
			return (1/(2*n)) * sum_xi(X,n,x)
		return inner

	return Estimador(
		name = "gamma",
		vero = vero,
		momento = momento
	)


def getLogNormal():
	def vero(*X):
		def ln2(xi):
			return math.pow(math.log(xi),2)

		def inner(x):
			n = len(x)
			somatorio = sum_xi(X,n,x, transform = ln2)
			return somatorio/n


	def momento(*X):
		def inner(x):
			n = len(X)
			somatorio_1_a_n = sum_xi(X,n,x)
			a = math.log(1/n)
			b = math.log(somatorio_1_a_n)
			return math.pow(2*(a + b), 0.5)
		return inner

	return Estimador(
		name = "logNormal",
		vero = vero,
		momento = momento
	)


def getExponencial():
	def vero(*X):
		def inner(x):
			n = len(x)
			somatorio_1_a_n = sum_xi(X,n,x, transform = None)
			return n / (somatorio_1_a_n)

	def momento(*X):
		def inner(x):
			n = len(X)
			somatorio_1_a_n = sum_xi(X,n,x)
			return n / somatorio_1_a_n
		return inner

	return Estimador(
		name = "exponencial",
		vero = vero,
		momento = momento
	)

class PegaEstimadores():
	def __init__(self):
		self.gamma = getGamma()
		self.exponencial = getExponencial()
		self.logNormal = getLogNormal()

	def values(self):
		return [self.gamma, self.exponencial, self.logNormal]

name = "Questão 1"
def main():
	return runFunctions([
		getGamma,
		getExponencial,
		getLogNormal,
	])