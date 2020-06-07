import math
from os import path
from src.utils import chalk, loadSeries, loadCSV, runFunctions, Log, Filer
from src.utils.estimators import Estimator
from src.utils import constants
from src import Q1
import numpy as np
import random
from matplotlib import pyplot as plt
from probscale import plot_pos, probplot

name = "Questão 4"
log = Log().instance(prepend = chalk.cyan("Q4"))
curr_location = path.join('.','src', 'Q4')

A1 = loadSeries("aluno3_A1.csv")

class TestValues:
	def __init__(self, repetitions = 1000, chunkLimitSize = 4000, chunkSizeStepIncrease = 1):
		"""
			Initialize the test values

			- repetitions: number of times a macro test is repeated
			- chunkLimitSize: the max limit size of a chunk
			- chunkSizeStepIncrease: the amount that the chunkSize should increase on every micro test
		"""
		self.repetitions = 1000
		self.chunkLimitSize = 4000
		self.chunkSizeStepIncrease = 1
		self._debugMode = False
		self.name = ""

	def init(self, name: str):
		"""
			Updates the outputFile

			@param {str} name: the name of the output file to store the test results
		"""
		self.name = name
		if self._debugMode:
			self.name = f"{name}_debug"
		self.updateGamma()
		return self

	def savefig(self, plot):
		Filer.savefig(plot, self.name)

	def showConsole(self):
		return self._debugMode

	def setDebugMode(self):
		"""
			Sets debug mode

			Sets the test properties for values that should make the test very imprecise, but run fast
		"""

		xlog = log.instance().indentMore()
		xlog.split().skip(2)
		xlog.warn("debug mode is set to " + chalk.lightred("True"))
		xlog.skip(2).split()

		self._debugMode = True
		self.repetitions = 10
		self.chunkLimitSize = 50
		self.chunkSizeStepIncrease = 5

		return self

	def updateGamma(self):
		gamma.sizes = np.arange(1,min(A1.length,self.chunkLimitSize), self.chunkSizeStepIncrease)
		return self

vals = TestValues()

class UseChunks:
	"""
		Class to help work with chunks of every.

		This class should split the input array (aluno3_A1.csv's data) and split into multiple arrays
		where each array should be equally splitted into [K] number of chunks of sizes [n/K], where [n]
		is the number of samples in the input array.

		The array of chunks should be interpreted as an array of sub-samples.
	"""

	def __init__(self, verbose = False):
		"""
			Initializes the UseChunks class

			- sizes: an array of chunkSize to be used on every test,
			essencially, this is an array with the [n/K] values
		"""
		self.verbose = verbose
		self.sizes = np.arange(1,min(A1.length,vals.chunkSizeStepIncrease), 1)
		self.__chunkSizes = {}

	def resetChunkSize(self):
		"""
			Reset the unique array of chunkSizes
		"""
		self.__chunkSizes = {}
		return self

	def getChunkSizes(self):
		"""
			Get an array of chunkSizes for every array of chunks
		"""
		return [item for item in self.__chunkSizes.values()]

	def forAllChunks(self, forChunks):
		"""
			For every pre-defined chunkSize, generates an array of chunks and
			seed to forChunks as argument, the array of chunks

			@param {forChunks}
		"""
		sizes = self.sizes

		for n_K in sizes:
			K = A1.length/(max(n_K, 1))

			np.random.shuffle(A1.values)
			chunks = np.array_split(A1.values, K)

			chunksKey = str(len(chunks[0]))
			if chunksKey in self.__chunkSizes:
				continue
			self.__chunkSizes[chunksKey] = chunksKey


			forChunks(chunks)

	def forAllRandomChunks(self, forChunk):
		"""
			For every pre-defined chunkSize, generates an array of chunks and
			seed to forChunk as argument, a random chunk in that array

			@param {forChunk}
		"""
		sizes = self.sizes

		for n_K in sizes:
			if n_K <= 0:
				continue
			np.random.shuffle(A1.values)
			chunk = A1.values[:n_K]

			self.__chunkSizes[str(n_K)] = n_K

			forChunk(chunk, n_K)

class Gamma(UseChunks):
	def __init__(self):
		super().__init__(verbose = True)
		self.gamma = Q1.Gamma()
		self.gamma.setMethod(constants.EstimatorMethod.likelihood)
		self.estimator = self.gamma.estimator


	def empiricalEvidenceBias(self):
		"""
			Calculate the Empirical Evidence of the Bias ("Viés")
		"""
		name = "Bias"
		vals.init(name)

		xlog = log.instance().indentMore()

		bias_results = []

		# The population parameter used is the Estimated value for the entire dataset
		self.parameter = self.estimator.getEstimative(A1.values)

		def forChunks(chunks):
			"""
				Function to be ran on every array of chunks generated.

				Where the chunks is an array of sub-samples of the population.
			"""
			chunksResults = []
			for chunk in chunks:
				"""
					Function to be ran on every of chunk (sub-sample) generated.

					Here we calculate:
						- the SampleEstimative of the chunks, using the estimator.
						- the EstimationError = (Estimated value for the chunk) - Parameter
				"""
				sample_estimative = self.estimator.getEstimative(chunk)
				estimation_error = sample_estimative - self.parameter

				# Store the chunk result
				chunksResults.append(estimation_error)
				
			# Get the mean value of the chunks results
			bias_estimative = np.mean(chunksResults)

			K = len(chunks[0])
			n_K = len(chunks)

			# Store the result
			bias_results.append((bias_estimative, K, n_K))

		self.forAllChunks(forChunks)
	

		xlog = log.instance().indentMore(1)
		xlog.print("Bias results")
		joiner = "  "
		title = joiner.join(["{:^8}".format("K"),"{:^6}".format("n/K"),"{:^16}".format("Bias Estimative")])
		
		def writeResults(file):
			if vals.showConsole():
				xlog.print(title)
			file.write(title + "\n")

			for (bias_estimative, K, n_K) in bias_results:
				formatted_bias = format(bias_estimative, ".3e")

				if bias_estimative >= 0:
					formatted_bias = " " + formatted_bias
				fBias, fn_K, fK = "{:^4}".format(n_K), "{:2d}".format(K), formatted_bias
				file.write(joiner.join(["  {}  ".format(x) for x in [fBias, fn_K, fK]]))
				file.write("\n")
				if vals.showConsole():
					xlog.print(joiner.join([chalk.cyan(x) for x in [fBias, fn_K, fK]]))


		Filer.writeTXT(vals.name, writeResults)
		

	def empiricalEvidenceMSE(self):
		"""
			Calculate the Empirical Evidence of the MSE (Mean Squared Error)
		"""
		name = "MSE"
		vals.init(name)
		self.resetChunkSize()

		self.results = []

		# The population parameter used is the Estimated value for the entire dataset
		self.parameter = self.estimator.getEstimative(A1.values)

		microExperimentResult = []
		
		def forChunk(chunk, chunkSize):
			"""
				Function to be ran on every array of chunks generated.

				Where the chunk is random sample an array of sub-samples of the population.


				Here we calculate:
					- the TheoricMSE: [((Population's Parameter)^2) / ( 2 * (n/K) )]
					- the SampleEstimative of the chunks, using the estimator.
					- the SquaredError: [( SampleEstimative - Population's Parameter) ^ 2]
			"""
			theoricMSE = self.parameter ** 2/(2*chunkSize)

			sample_estimative = self.estimator.getEstimative(chunk)
			squaredError = (sample_estimative - self.parameter) ** 2

			microExperimentResult.append(squaredError)

		experiments = []
		for rep in range(vals.repetitions):
			"""
				Reruns the microExperiment M number of times
			"""
			microExperimentResult = []

			self.forAllRandomChunks(forChunk)
			experiments.append(microExperimentResult)

		# Array of meanSquaredError results for every column in the experiments matrix
		experimentalMeanSquaredErrors = [np.mean(row) for row in np.transpose(experiments)]

		chunkSizes = self.getChunkSizes()
		y1 = experimentalMeanSquaredErrors
		plt.plot(chunkSizes, y1, label = "MSE Experimental")

		y2 = [self.parameter ** 2/(2*chunkSize) for chunkSize in chunkSizes]
		plt.plot(chunkSizes,y2, label = "MSE Teórico")

		plt.ylabel("Erro Médio Quadrático Estimado")
		plt.xlabel("Tamanho de Sub-Amostra")
		plt.legend()
		vals.savefig(plt)
		plt.close()
		
		
	def empiricalEvidenceConsistency(self):
		"""
			Calculate the Empirical Evidence of the Consistency
		"""
		name = "Consistency"

		vals.init(name)

		# Use the Likelihood method for estimation
		self.estimator = self.gamma.setMethod(constants.EstimatorMethod.likelihood).estimator

		# The population parameter used is the Estimated value for the entire dataset
		self.parameter = self.estimator.getEstimative(A1.values)

		plt.figure()

		plotColors = ["r-", "b-", "g-", "c-"]
		epsilons = [1, 0.1, 0.01, 0.001]
		for index in range(len(epsilons)):
			"""
				Rerun the entire macro-experiment for different values of epsilon.
			"""

			epsilon = epsilons[index]
			self.resetChunkSize()

			xlog = log.instance().indentMore()

			microExperimentResult = []
			experiments = []

			def forRandomChunk(chunk, chunkSize):
				"""
					Function to be ran on every array of chunks generated.

					Where the chunk is random sample an array of sub-samples of the population.

					Here we calculate:
						- the SampleEstimative of the chunks, using the estimator.
						- the EstimationError = (Estimated value for the chunk) - Parameter
						- the absolute value of difference between the [ EstimationError ] and [ Epsilon ],
						which will called [abs_diff]

					Than a filter is applied on [abs_diff], where if [abs_diff] is smalled than [epsilon], returns 1
					otherwise, returns 0
				"""
				estimative = self.estimator.getEstimative(chunk)

				abs_diff = math.fabs(estimative - self.parameter)

				microExperimentResult.append(1 if abs_diff < epsilon else 0)

			for rep in range(vals.repetitions):
				"""
					Reruns the microExperiment M number of times
				"""
				microExperimentResult = []

				self.forAllRandomChunks(forRandomChunk)
				experiments.append(microExperimentResult)

			# Array of estimatedProbabilities results for every column in the experiments matrix
			estimatedProbability = [np.mean(row) for row in np.transpose(experiments)]

			y1 = estimatedProbability
			plt.plot(self.getChunkSizes(), y1, plotColors[index], label = f"ε = {str(epsilon)}")

			plt.ylabel("Probabilidade Estimada")
			plt.xlabel("Tamanho de Sub-Amostra")
			plt.legend()

		# Set transparent dots in probability plot
		ax = plt.gca()
		for line in ax.get_lines():
			line.set_alpha(0.5)

		vals.savefig(plt)
		plt.close()


gamma = Gamma()

def main():
	Filer.setLocation("Q4")
	vals.setDebugMode()

	exitCode = runFunctions([
		gamma.empiricalEvidenceBias,
		gamma.empiricalEvidenceMSE,
		gamma.empiricalEvidenceConsistency
	], timer = True)
	return exitCode > 0