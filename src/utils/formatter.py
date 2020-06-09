
class Format:

	@staticmethod
	def percent(value, digits = 4):
		f = "{:." + str(digits) + "}"
		return f.format(value)

	def e(value, digits = 4):
		return format(value, f".{digits}e")