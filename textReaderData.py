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
my_data = TextReaderData('MNIST/emnist-balanced-train-images-idx3-ubyte',
	'MNIST/emnist-balanced-train-labels-idx1-ubyte')
example = 31

skip = 5
print('Dimensions: %s x %s' % (my_data.X.shape[0], my_data.X.shape[1]))

for i in range(len(my_data.y)):
	if my_data.y[i] == example:
		skip = skip - 1
		if skip < 0:
			print('\nClassification', my_data.y[i])
			sys.stdout.flush()
			img = Image.fromarray(my_data.X[i].reshape(28, 28).astype('uint8'), 'L')
			img.show()
			break
"""