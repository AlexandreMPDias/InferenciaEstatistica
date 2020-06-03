from matplotlib import pyplot
from probscale import plot_pos, probplot
from scipy import stats
import seaborn
from src.utils import chalk, loadSeries, loadCSV, runFunction
from os import path

from src.Q2.plotter import plotter

A1 = loadSeries("aluno3_A1.csv")
A2 = loadSeries("aluno3_A2.csv")
A3 = loadSeries("aluno3_A3.csv")

name = "Questão 2"

class Plotter:
	def __init__(self):
		self.plotCache = []

		self.config = dict(
			probax='y',
			datascale='log',
			plottype='prob',
			problabel="Probabilidade",
			scatter_kws=dict(marker='.', linestyle='none')
		)

	def setName(self, name):
		self.name = name
		self.config["datalabel"] = name
		self.output = path.join("src","Q2", "plots", name + ".png")

	def setData(self, data):
		self.data = data

	def save(self):
		import matplotlib.pyplot as curr_plot

		fig = probplot(
			self.data,
			**self.config
		)

		# axis.set_xlim(left=1, right=100)
		# axis.set_ylim(bottom=0.13, top=99.87)

		fig.tight_layout()
		# curr_plot.savefig(self.output, format="png")

		pyplot.subplot(211)
		self.plotCache.append(fig)

	def show(self):

		pyplot.show()

plot = Plotter()

def mock():
	clear_bkgd = {'axes.facecolor':'none', 'figure.facecolor':'none'}
	seaborn.set(style='ticks', context='talk', color_codes=True, rc=clear_bkgd)

	tips = seaborn.load_dataset("tips")

	data = tips['total_bill']

	plot.setName("Teste")
	plot.setData(data)
	plot.save()

	seaborn.despine()


def load():
	pass


def probplotx():

	clear_bkgd = {'axes.facecolor':'none', 'figure.facecolor':'none'}
	seaborn.set(style='ticks', context='talk', color_codes=True, rc=clear_bkgd)

	# load up some example data from the seaborn package
	tips = seaborn.load_dataset("tips")
	iris = seaborn.load_dataset("iris")

	fig, ax3 = pyplot.subplots(figsize=(9, 6), ncols=1, sharex=True)
	common_opts = dict(
		probax='y',
		datascale='log',
		datalabel='Total Bill (USD)',
		scatter_kws=dict(marker='.', linestyle='none')
	)



	data = tips['total_bill']

	print(data)

	fig = probplot(data, ax=ax3, plottype='prob',
							problabel='Standard Normal Probabilities',  **common_opts)

	ax3.set_xlim(left=1, right=100)
	ax3.set_ylim(bottom=0.13, top=99.87)
	seaborn.despine()
	fig.tight_layout()

	pyplot.show()



def main():
	exitCode = 0

	functions = [
		mock,
		mock,
		mock,
		mock,
	]

	for function in functions:
		exitCode = exitCode + runFunction(function)

	plot.show()

	return exitCode > 0