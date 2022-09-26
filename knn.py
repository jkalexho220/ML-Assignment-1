import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from MLAlgorithm import MLAlgorithm

class KNN(MLAlgorithm):
	def __init__(self, n_neighbors=5):
		self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

	def plot_hyperparameter(self, X, y, testX, testy, name):
		parameterMaximum = 30
		fig, axes = plt.subplots(1, 1, figsize=(5, 5))
		accuracy = np.zeros(parameterMaximum)
		leafCount = np.arange(parameterMaximum) + 1
		for i in range(parameterMaximum):
			self.model = KNeighborsClassifier(n_neighbors=i+1)
			self.learn(X, y)
			result = self.solve(testX)
			accuracy[i] = np.sum([1 for element in np.subtract(result, testy) if element == 0]) / testy.shape[0]
		axes.grid()
		axes.plot(leafCount, accuracy, "o-")
		axes.set_xlabel("Number of Neighbors")
		axes.set_ylabel("Accuracy")
		axes.set_title("KNN for " + name + " Data")
