import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import time

# custom classes
from esportsData import ESportsData
from textReaderData import TextReaderData
from decisionTree import DecisionTree
from knn import KNN
import sciplotter

def log(info):
	print(info)
	sys.stdout.flush()

def testClassifier(algorithm, X, y):
	total = len(y)
	incorrect = []
	results = algorithm.solve(X)
	for i in range(total):
		if results[i] != y[i]:
			incorrect.append((results[i], y[i]))

	correct = total - len(incorrect)
	log("Total: " + str(total) + " Correct: " + str(correct) + " Accuracy: " + str(float(correct)/float(total)))

log("Reading datasets...")


text_training_data = TextReaderData('MNIST/emnist-balanced-train-images-idx3-ubyte',
	'MNIST/emnist-balanced-train-labels-idx1-ubyte')

text_test_data = TextReaderData('MNIST/emnist-balanced-test-images-idx3-ubyte',
	'MNIST/emnist-balanced-test-labels-idx1-ubyte')

esports_training_data = ESportsData()

esports_training_data.readData()

esports_test_data = esports_training_data.partitionData(0.2)

classifier = KNN()

log("Read done!")

#classifier.learn(text_training_data.X, text_training_data.y)

#classifier.plot_hyperparameter(text_training_data.X, text_training_data.y, text_test_data.X, text_test_data.y)
#classifier.plot_hyperparameter(esports_training_data.X, esports_training_data.y, esports_test_data.X, esports_test_data.y)

#classifier.plot_learning_curve("KNN Text Data", text_training_data.X, text_training_data.y)
classifier.plot_learning_curve("KNN ESports", esports_training_data.X, esports_training_data.y)

plt.show()


"""
log("Training classifier...")
start = time.monotonic()
classifier.learn(training_data.X, training_data.y)
end = time.monotonic()
log("Training done! Total elapsed: " + str(end - start))
log("Testing classifier...")
start = time.monotonic()
testClassifier(classifier, test_data.X, test_data.y)
end = time.monotonic()
log("Testing complete! Total elapsed: " + str(end - start))
"""