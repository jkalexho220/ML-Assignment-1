import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from MLAlgorithm import MLAlgorithm

class Boost(MLAlgorithm):
	def __init__(self):
		self.model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100)

	def plot_hyperparameter(self, X, y, testX, testy, name):
		parameterMaximum = 5
		fig, axes = plt.subplots(1, 1, figsize=(5, 5))
		accuracy = np.zeros(parameterMaximum)
		parameterCount = (np.arange(parameterMaximum) + 1) * 20
		for i in range(parameterMaximum):
			self.model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=20*(i+1))
			self.learn(X, y)
			result = self.solve(testX)
			accuracy[i] = np.sum([1 for element in np.subtract(result, testy) if element == 0]) / testy.shape[0]
		axes.grid()
		axes.plot(parameterCount, accuracy, "o-")
		axes.set_xlabel("Number of Estimators")
		axes.set_ylabel("Accuracy")
		axes.set_title("Boosting for " + name + " Data")
