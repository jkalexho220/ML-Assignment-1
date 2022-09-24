import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from MLAlgorithm import MLAlgorithm

class DecisionTree(MLAlgorithm):
	def __init__(self, criterion="gini", splitter="best"):
		self.model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_leaf=10)

	def plot_hyperparameter(self, X, y, testX, testy):
		parameterMaximum = 30
		fig, axes = plt.subplots(1, 1, figsize=(5, 5))
		accuracy = np.zeros(parameterMaximum)
		leafCount = np.arange(parameterMaximum) + 1
		for i in range(parameterMaximum):
			#self.model = DecisionTreeClassifier(min_samples_leaf=i+1)
			self.model = DecisionTreeClassifier(max_depth=i+1)
			self.learn(X, y)
			result = self.solve(testX)
			accuracy[i] = np.sum([1 for element in np.subtract(result, testy) if element == 0]) / testy.shape[0]
		axes.grid()
		axes.plot(leafCount, accuracy, "o-")
		axes.set_xlabel("Maximum Depth")
		axes.set_ylabel("Accuracy")
		axes.set_title("Maximum Depth for ESports Data")
