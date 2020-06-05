from matplotlib import pyplot
from probscale import plot_pos, probplot
from scipy import stats

def rightMerge(*dicts):
	out = {}
	print(dicts)
	for sdict in dicts:
		for (key,values) in sdict.items():
			out[key] = values
	return out

class _ProbPlotter:
	def __init__(self):
		self.datas = []

		self.common = dict(
			probax='y',
			datascale='log',
			scatter_kws=dict(marker='.', linestyle='none')
		)

	def add(self, name, values, **plot_args):
		self.datas.append(
			dict(
				values = values,
				datalabel = name,
				plot_args = rightMerge(self.common, plot_args)
			)
		)

		print(f"{name} was added")


	def plot(self):
		fig, axs = pyplot.subplots(figsize=(9, 6), ncols=3, sharex=True)

		datas = []
		for i in range(0, 2):
			d = self.datas[i]
			d['plot_args']['ax'] = axs[i]
			datas.append(d)

		for data in datas:
			fig = probplot(
				data['values'],
				plottype='pp',
				problabel='Probabilidade',
				**data['plot_args']
			)

		axs[2].set_xlim(left=1, right=100)
		axs[2].set_ylim(bottom=0.13, top=99.87)
		fig.tight_layout()

		print(f"plotted")

		pyplot.show()

ProbPlotter = _ProbPlotter()