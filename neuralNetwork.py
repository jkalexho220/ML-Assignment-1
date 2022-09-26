import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.neural_network import MLPClassifier
from MLAlgorithm import MLAlgorithm

class NeuralNetwork(MLAlgorithm):
	def __init__(self):
		self.model = MLPClassifier(hidden_layer_sizes=(30,30))

	def plot_hyperparameter(self, X, y, testX, testy, name):
		parameterMaximum = 8
		fig, axes = plt.subplots(1, 1, figsize=(5, 5))
		accuracy = np.zeros(parameterMaximum)
		parameterCount = (np.arange(parameterMaximum) + 1) * 20
		for i in range(parameterMaximum):
			print("Begin learning " + str(i))
			sys.stdout.flush()
			self.model = MLPClassifier(hidden_layer_sizes=(20*(i+1),))
			self.learn(X, y)
			result = self.solve(testX)
			accuracy[i] = np.sum([1 for element in np.subtract(result, testy) if element == 0]) / testy.shape[0]
			print("Learned " + str(i))
			sys.stdout.flush()
		axes.grid()
		axes.plot(parameterCount, accuracy, "o-")
		axes.set_xlabel("Size of hidden layer 1")
		axes.set_ylabel("Accuracy")
		axes.set_title("Neural Network for " + name + " Data")
