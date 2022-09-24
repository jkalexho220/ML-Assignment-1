"""
import sys
import os
from PIL import Image
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

from mlxtend.data import loadlocal_mnist

class TextReaderData:
	def __init__(self, data=None, labels=None):
		self.X = np.empty([0,784], dtype=np.int8)
		self.y = np.empty([0,], dtype=np.int8)
		if data and labels:
			self.readData(data, labels)

	def readData(self, data, labels):
		tempX, tempy = loadlocal_mnist(images_path=data, labels_path=labels)
		self.X = np.concatenate((self.X, tempX))
		self.y = np.concatenate((self.y, tempy))

#X, y = loadlocal_mnist(images_path='MNIST/emnist-balanced-train-images-idx3-ubyte', labels_path='MNIST/emnist-balanced-train-labels-idx1-ubyte')



"""
example = 24

skip = 3
print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))

for i in range(len(y)):
	if y[i] == example:
		skip = skip - 1
		if skip < 0:
			print('\nArray', X[i])
			print('\nClassification', y[i])
			sys.stdout.flush()
			img = Image.fromarray(X[i].reshape(28, 28), 'L')
			img.show()
			break
"""