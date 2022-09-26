import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from sklearn.svm import SVC
from MLAlgorithm import MLAlgorithm

class SupportVectorMachine(MLAlgorithm):
	def __init__(self):
		self.model = SVC()

	def plot_hyperparameter(self, X, y, testX, testy, name):
		print(name)
		for i in ['linear', 'poly', 'rbf', 'sigmoid']:
			print("Begin learning " + i)
			sys.stdout.flush()
			self.model = SVC(kernel=i)

			startTime = time.monotonic()
			self.learn(X, y)
			learnTime = time.monotonic()

			result = self.solve(testX)
			solveTime = time.monotonic()

			accuracy = np.sum([1 for element in np.subtract(result, testy) if element == 0]) / testy.shape[0]
			print("Learned " + i + ". Accuracy: " + accuracy + ". Learn Time: " + str(learnTime - startTime) + ". Solve Time: " + str(solveTime - learnTime))
			sys.stdout.flush()
