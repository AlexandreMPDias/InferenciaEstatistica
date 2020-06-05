from matplotlib import pyplot
from probscale import plot_pos, probplot
from scipy import stats
from src.utils import chalk, loadSeries, loadCSV, runFunctions
from os import path

from src.Q1 import PegaEstimadores

A1 = loadSeries("aluno3_A1.csv")
A2 = loadSeries("aluno3_A2.csv")
A3 = loadSeries("aluno3_A3.csv")

name = "Questão 2"

def mostra_grafico():
	from src.Q2.probplotter import ProbPlotter
	estimadores = PegaEstimadores()
	for data in [A1]:
		for estimador in estimadores.values():
			def apply(x):
				return estimador.vero(lambda x: x)

			ProbPlotter.add(f"{data.name}_estimador_{estimador.name}_vero", [apply(x) for x in data.values])
	ProbPlotter.plot()
		
	pass


def main():
	exitCode = runFunctions([
		mostra_grafico
	])
	return exitCode > 0