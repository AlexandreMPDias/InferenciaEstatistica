from src.utils.debug import chalk
chalk.wrappers = ["{ "," }"]

import src.Q1 as Q1
import src.Q2 as Q2
import src.Q3 as Q3
import src.Q4 as Q4
import src.Q5 as Q5

"""
	Defina aqui quais questões serão rodadas.

	Para rodar todar:
		questionsToRun = [Q1, Q2, Q3, Q4, Q5]

	Para rodar apenas a [Questão 1]
		questionsToRun = [Q1]
"""
questionsToRun = [Q2] #[Q1, Q2, Q3, Q4, Q5]

def runSingle(question):
	qName = chalk.cyan(question.name)
	print(qName + " - rodando")
	if(question.main()):
		print(qName + " - " + chalk.lightred("finalizada com erro", False))
	else:
		print(qName + " - " + chalk.lightgreen("finalizada com sucesso", False))

for question in questionsToRun:
	runSingle(question)
	print("\n")