class EstimatorMethod:
	likelihood = "Likelihood"
	moment = "Moment"

	options = []
EstimatorMethod.options = [EstimatorMethod.likelihood, EstimatorMethod.moment]

class Density:
	gamma = "Gamma"
	exponential = "Exponential"
	logNormal = "Log-Normal"

	options = []

	@staticmethod
	def parametersOf(density):
		if (density == "Gamma"):
			return [Parameter.theta]
		elif (density == "Exponential"):
			return [Parameter.lamb]
		elif (density == "Log-Normal"):
			return [Parameter.sigma]

Density.options = [Density.gamma, Density.exponential, Density.logNormal]

class Parameter:
	sigma = "Sigma"
	theta = "Theta"
	lamb = "Lambda"
	options = []

Parameter.options = [Parameter.sigma, Parameter.theta, Parameter.lamb]