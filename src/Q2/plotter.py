import matplotlib.pyplot as plt

class Plotter:
	def __init__(self):
		self.config = {}

	def setName(self, name):
		self.name = name

	def setData(self, data):
		self.data

	def setConfig(self, **kwargs):
		self.config = kwargs

	def build(self):
		plt.plot(self.data)

		for key in self.config:
			if(key == "legend"):
				plt.legend(self.config["legend"], loc = "upper left")
			else:
				plt[key](self.config[key])


plotter = Plotter()