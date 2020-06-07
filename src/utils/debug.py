"""

DEBUG.PY

Used for code debugging.
Contains the following classes:
[** = not yet implemented]
		Log:
			error(strging)		   : returns self
			success(strging)		  : returns self
			warn(strging)			: returns self
			info(strging)			: returns self
			color(Color, strging)	: returns self
			split(Color)			: returns self
		**critical(strging)	  : returns self

		IntegrityTest:
			isValidProcess(Process, strging)	  : returns True/False
			isCameraConnected(Boolean)		   : returns True/False
			canLocateFile(strging, strging)		: returns True/False
			canLocateDir(strging, strging)		 : returns True/False
		**isNone(Object, strging)			 : returns self
		**performInitialChecks()			 : returns self
			hasInternetConnection() [not tested] : returns self



Requires installation of colorama: [ pip install colorama ]

Note:
	-- Not designed to be Thread safe.
	-- Debug doesnt work yet

"""
try:
	from colorama import init as _color_init
	from colorama import Fore, Back, Style
	_color_init()

	FONT_COLOR_RED = Fore.RESET+Fore.RED
	FONT_COLOR_GREEN = Fore.RESET+Fore.GREEN
	FONT_COLOR_RESET = Fore.RESET
	FONT_COLOR_YELLOW = Fore.RESET+Fore.YELLOW

	"""
	Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
	Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
	Style: DIM, NORMAL, BRIGHT, RESET_ALL
	"""
except:
	print('Colorama not installed. Colors won\'t work')
	FONT_COLOR_RED = ''
	FONT_COLOR_GREEN = ''
	FONT_COLOR_RESET = ''
	FONT_COLOR_YELOW = ''

# Debug Levels
LOG_SHOW_NOTHING = -1
LOG_SHOW_ERROR = 0
LOG_SHOW_WARNING = 1
LOG_SHOW_SUCCESS = 2
LOG_SHOW_INFO = 3
import sys


class Log():
	def __init__(self, debug_level=LOG_SHOW_INFO):
		self._debug_level = debug_level
		self.writer = sys.stdout.write
		self.indentation = 0
		self.beforeEach = ""
		self.afterEach = "\n"

	def instance(self, prepend: str = "", append: str = "\n"):
		_log = Log(self._debug_level)
		b4Each =_log.beforeEach if prepend == "" else f"{_log.beforeEach}{prepend} "
		_log.beforeEach = b4Each
		_log.afterEach = append
		return _log

	def indentMore(self, amount = 1):
		self.indentation = self.indentation + amount
		return self

	def indentLess(self, amount = 1):
		nextIndentation = self.indentation - amount
		self.indentation = 0 if nextIndentation <= 0 else nextIndentation
		return self

	def resetIndent(self):
		self.indentation = 0
		return self

	def print(self, strg: str):
		self.__print(strg)
		return self

	def error(self,strg):
		"""
			Prints an error message on the console. [Error] colored RED
		"""
		if self._debug_level >= LOG_SHOW_ERROR:
			self.__print(FONT_COLOR_RED + '[ Error ]: ' + FONT_COLOR_RESET + str(strg))
		return self


	def success(self,strg):
		"""
			Prints a success message on the console. [success] colored Green
		"""
		if self._debug_level >= LOG_SHOW_SUCCESS:
			self.__print(FONT_COLOR_GREEN + '[Success]: ' + FONT_COLOR_RESET + str(strg))
		return self

	def info(self,strg):
		"""
			Prints an info message on the console. [Info] colored WHITE
		"""
		if self._debug_level >= LOG_SHOW_INFO:
			self.__print('[ Info  ]: ' + str(strg))
		return self

	def warn(self,strg):
		"""
			Prints an Warning message on the console. [Warn] colored WHITE
		"""
		if self._debug_level >= LOG_SHOW_WARNING:
			self.__print(FONT_COLOR_YELLOW + '[Warning]: ' + FONT_COLOR_RESET + str(strg))
		return self

	def color(self,col="", strg=''):
		"""
			Prints a colored message on the console. The entire strging colored @param col
		"""
		if self._debug_level >= LOG_SHOW_INFO:
			self.__print(col + str(strg) + FONT_COLOR_RESET)
		return self

	def split(self,col=""):
		"""
			Draws a line on the console.
		"""
		if self._debug_level >= LOG_SHOW_INFO:
			self.color(col,"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
		return self

	def skip(self, skips = 1):
		if self._debug_level >= LOG_SHOW_WARNING:
			skip_str = ""
			for i in range(0,skips - 1):
				skip_str = skip_str + "\n"
			self.__print(skip_str)
		return self

	def catch(self, error, message = ""):
		"""
			Prints an exception error
		"""
		import traceback
		if self._debug_level >= LOG_SHOW_ERROR:
			if len(message) > 0:
				self.error(message).split()
			self.color(FONT_COLOR_RED,str(error))

			traceback.print_exception(Exception, error, None)

			self.split()
		return self

	def __print(self, strg):
		"""
			Prints a message on the console.
		"""
		indent = "\t" * self.indentation
		self.writer("\033[F\033[K\r" + indent + self.beforeEach + str(strg) + self.afterEach)

	def setAppendable(self, appendable):
		self.writer = appendable

class Attempt:
	def __init__(self, log = None):
		self.log = log if log is not None else Log()

	def run(self,key, action):
		return self.__createAttempt(key, action, ['running', 'success'])

	def load(self,key, action):
		return self.__createAttempt(key, action, ['loading', 'loaded'])

	def __createAttempt(self, key, action, statusMethods):
		status = Status(key)
		status.method = sys.stdout.write
		ret = None
		getattr(status, statusMethods[0])()
		try:
			ret = action()
			status.method = print
			getattr(status, statusMethods[1])()
		except Exception as e:
			status.failed()
			raise e
		return ret

	def custom(self, key, action, statusMethods):
		status = Status(key)
		ret = None
		status.custom(chalk.white, statusMethods[0])
		try:
			ret = action()
			status.custom(chalk.green, statusMethods[1],True)
		except Exception as e:
			status.failed()
			raise e
		return ret

	def __resolveStatusMethod(self, status, met):
		if type(met) is str:
			getattr(status, met)()
		else:
			status.custom(met[0], met[1], met[2])
		

class __Chalk:
	def __init__(self):
		self.wrappers = ["{","}"]

	def resetWrapper(self):
		self.wrappers = ["{","}"]

	def green(self,strg, withBrackets = True):
		return self.__parse(Fore.GREEN, strg, withBrackets)

	def red(self,strg, withBrackets = True):
		return self.__parse(Fore.RED, strg, withBrackets)

	def yellow(self,strg, withBrackets = True):
		return self.__parse(Fore.YELLOW, strg, withBrackets)

	def blue(self,strg, withBrackets = True):
		return self.__parse(Fore.BLUE, strg, withBrackets)

	def cyan(self,strg, withBrackets = True):
		return self.__parse(Fore.CYAN, strg, withBrackets)

	def lightgreen(self,strg, withBrackets = True):
		return self.__parse(Fore.LIGHTGREEN_EX, strg, withBrackets)

	def lightred(self,strg, withBrackets = True):
		return self.__parse(Fore.LIGHTRED_EX, strg, withBrackets)

	def magenta(self, strg, withBrackets = True):
		return self.__parse(Fore.MAGENTA, strg, withBrackets)

	def white(self,strg, withBrackets = True):
		return self.__parse(Fore.WHITE, strg, withBrackets)

	def __parse(self, foreColor, strg, withBrackets):
		if(withBrackets):
			b4, after = self.wrappers
			return f"{foreColor}{b4}{str(strg)}{after}{FONT_COLOR_RESET}"
		else:
			return f"{foreColor}{str(strg)}{FONT_COLOR_RESET}"

chalk = __Chalk()

class Status:
	def __init__(self, message):
		self.message = message
		self.method = print

	def loading(self, deleteLine = False):
		message = chalk.white(f"[ {self.message} ]: loading", False)
		sys.stdout.write(message)
		return self

	def loaded(self, deleteLine = True):
		self.custom(chalk.green, "loaded", deleteLine)
		return self

	def running(self, deleteLine = False):
		message = chalk.white(f"[ {self.message} ]: running", False)
		sys.stdout.write(message)

	def skipped(self, deleteLine = False):
		self.custom(chalk.yellow, "skipped", deleteLine)
		return self

	def waiting(self, deleteLine = False):
		self.custom(chalk.yellow, "waiting", deleteLine)
		return self

	def success(self, deleteLine = True):
		self.custom(chalk.green, "success", deleteLine)
		return self

	def failed(self, deleteLine = True):
		self.custom(chalk.red, "failed", deleteLine)
		return self

	def writing(self, deleteLine = True):
		self.custom(chalk.lightgreen, "writing", deleteLine)
		return self

	def custom(self, colorMethod, key, deleteLine = False):
		message = colorMethod(f"[ {self.message} ]: {key}", False)
		if deleteLine:
			sys.stdout.write("\033[F\r")
			sys.stdout.write("\033[K\r")
		self.method(message)

log = Log()
# try:
# 	import httplib
# except:
# 	import http.client as httplib
import pathlib
import os

class IntegrityTest():
	def __init__(self):
		pass

	def canLocateDir(self,path, createIfNotFound=False, hint=''):
		"""
			Check if the path in the parameter { path } is a directory.
		@param createIfNotFound: is set to True, and the path does not point to
			a directory, the directory will be created.
		"""
		found = self.__locator(path, os.path.isdir(path),hint)
		if not found and createIfNotFound:
			pathlib.Path(path).mkdir(parents=True, exist_ok=True)
			log.success('[ {} ] created.'.format(path))
		return found

	def canLocateFile(self,file, hint=''):
		"""
		Check if the path in the parameter { file } is a file.
		"""
		return self.__locator(file, os.path.isfile(file),hint)

	def isCameraConected(self, PopUpOn=False):
		"""
		As the name suggest, checks if the camera is connected by trying to access the camera
		and by capturing any exceptions. If everything works fine, the camera is release and
		can be accessed normally.
		"""
		try:
			import cv2
			cap = cv2.VideoCapture(0)
			ret, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			cap.release()
			log.success('Camera found')
			return True
		except:
			log.error('Camera not found.')
			raise

	def isValidProcess(self,th, name=''):
		"""
		Check if a given Process [th] is a valid(not null) Process variable
		returns if not. Then try to start the Process.
	@param th   : The Process variable
	@param name : The local name for the variable
		"""
		if th is None:
			if len(name) > 0:
				log.error(name + ' is not a valid Process')
			return
		if len(name) > 0:
			log.success('['+ name + '] is a valid Process')
			pass
		try:
			th.start()
			log.split().success(name+ ' successfully started.')
		except TypeError as te:
			log.error(name+' failed to start. TypeError').split()
			log.color(FONT_COLOR_RED,'['+name+'] ' + str(te))
		except EOFError as eof:
			log.split().error(name+ 'failed to start. EOFError')
		except OSError as ae:
			log.split().error(name+' failed to start. OSError')

	def hasInternetConnection(self):
		"""
		Check there's an internet connection
		"""
		return True
	# if self.__have_internet():
	# 	log.success('Internet Connection found')
	# 	return True
	# else:
	# 	log.error('No Internet Connection')
	# 	return False
		pass

	def performInitialChecks(self):
		"""
		Perfoms initial checks.
	[NOT YET IMPLEMENTED]
		"""
		self.hasInternetConnection()

	# def __have_internet(self):
	# 	"""
	# 	Checks if it is possible to connect to [www.google.com]
	# 	"""
	# 	conn = httplib.HTTPConnection("www.google.com", timeout=5)
	# 	try:
	# 		conn.request("HEAD", "/")
	# 		conn.close()
	# 		return True
	# 	except:
	# 		conn.close()
	# 		return False

	def __locator(self, path, found, hint):
		"""
		Prints a message if a path points to something.
		"""
		if found:
			log.success('[ {} ] found'.format(path))
			return True
		else:
			if not hint == '':
				log.error('[ {} ] not found\nHINT: {}'.format(path,hint))
			else:
				log.error('[ {} ] not found'.format(path))
			return False

import time
import numpy
round = numpy.round
# try:
# 	import numpy
# 	round = numpy.round
# except:
# 	import math
# 	round = math.round
class Perfomance:
	def __init__(self, dimension = 1000, decimals = 4):
		"""
			Initiates the Timer.
		"""
		if dimension < 1:
			print('Perfomance warning: Dimension set to a value smaller than 1, this wont increase precision')
		self.start = time.perf_counter()
		self.dimension = dimension
		self.decimals = decimals

	def elapsed(self):
		"""
			Return the elapsed time since the Timer was created or the last time it was resetted
			Default Dimension: Miliseconds
		"""
		return round((time.perf_counter() - self.start) * self.dimension, self.decimals)

	def elapsedStr(self):
		"""
			Return the elapsed time since the Timer was created or the last time it was resetted in String format
			Default Dimension: Miliseconds
		"""
		return f"[ { self.elapsed() } ]"

	def reset(self):
		"""
			Resets the Timer.
		"""
		self.start = time.perf_counter()
		return self
