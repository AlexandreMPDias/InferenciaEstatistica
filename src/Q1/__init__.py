from src.utils import chalk, loadSeries, loadCSV, runFunction

A1 = loadSeries("aluno3_A1.csv")
A2 = loadSeries("aluno3_A2.csv")
A3 = loadSeries("aluno3_A3.csv")

def estimador_vero_gamma():
	k = 2

def estimador_vero_exponencial():
	pass

def estimador_vero_logNormal():
	avg = 0
	pass

name = "Questão 1"
def main():
	exitCode = 0

	functions = [
		estimador_vero_gamma,
		estimador_vero_exponencial,
		estimador_vero_logNormal
	]

	for function in functions:
		exitCode = exitCode + runFunction(function)

	return exitCode > 0