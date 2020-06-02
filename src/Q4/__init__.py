from src.utils import chalk, loadSeries, loadCSV

A1 = loadSeries("aluno3_A1.csv")
A2 = loadSeries("aluno3_A2.csv")
A3 = loadSeries("aluno3_A3.csv")

name = "Questão 4"

def main():
	exitCode = 0

	functions = [
		
	]

	for function in functions:
		exitCode = exitCode + runFunction(function)

	return exitCode > 0