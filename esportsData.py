"""
import sys
import os
from PIL import Image
"""
import csv
import numpy as np
import scipy.io as sio

"""
Patch Number
Team 1 name
Team 2 name
Team 1 players
Team 2 players
Team 1 bans
Team 2 bans
Team 1 picks
Team 2 picks
"""
DATA_PATCH_NUMBER = 0
DATA_TEAM_NAMES = 1
DATA_CHAMP_NAMES = 0
#DATA_TEAM_1_NAME = 1
#DATA_TEAM_2_NAME = 2
#DATA_TEAM_1_PLAYERS = 3
#DATA_TEAM_1_BANS = 13
#DATA_TEAM_2_BANS = 18
#DATA_TEAM_1_PICKS = 23

READ_PATCH_NUMBER = 9
READ_WINNER = 24
READ_TEAM_NAME = 15
READ_CHAMPION = 17
READ_BANS = 18
#READ_PLAYER_NAME = 13

TEAM_1_PICK = 2
TEAM_1_BAN  = -1
TEAM_2_PICK = -2
TEAM_2_BAN  = 1

TEAM_1 = 1
TEAM_2 = -1

ESPORTS_DATA_LENGTH = 0

npStr = np.dtype(('U', 32))

def binarySearch(listName, name):
	point = len(listName) // 2
	jumpDist = (point + 1) // 2
	while jumpDist > 0:
		result = (name == listName[point])

		if result == 0:
			return point
		elif result > 0:
			point = point + jumpDist
		else:
			point = point - jumpDist
		jumpDist = (jumpDist + 1) // 2

	return -1


class ESportsData:
	def __init__(self):
		self.y = np.empty([0,], dtype=np.int8)
		self.files = ["League/2022.csv", "League/2021.csv", "League/2020.csv"]
		self.matchCount = 0
		self.teams = set()
		self.champions = set()
		for f in self.files:
			self.gatherData(f)
		ESPORTS_DATA_LENGTH = len(self.teams) + len(self.champions) + 1
		self.X = np.zeros([self.matchCount, ESPORTS_DATA_LENGTH], dtype=np.int8)
		self.matchCount = 0
		self.teams = list(self.teams).sort()
		self.champions = list(self.champions).sort()
		DATA_CHAMP_NAMES = DATA_TEAM_NAMES + len(self.teams)
		for f in self.files:
			self.readData(f)
		print(self.X[0])

	def gatherData(self, filename):
		with open(filename, newline='', encoding='UTF-8') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				self.teams.add(row[READ_TEAM_NAME])
				self.champions.add(row[READ_CHAMPION])
			self.matchCount = self.matchCount + reader.line_num - 1

	def readData(self, filename):
		readStep = 0
		with open(filename, newline='', encoding='UTF-8') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			gameID = "gameid"
			readStep = 11
			line = np.zeros(ESPORTS_DATA_LENGTH, dtype=np.int8)
			for row in reader:
				if readStep == 11:
					# read team 2 data
					line[binarySearch(self.teams, row[READ_TEAM_NAME]) + DATA_TEAM_NAMES] = TEAM_2
					for i in range(5):
						line[binarySearch(self.champions, row[READ_BANS + i]) + DATA_CHAMP_NAMES] = TEAM_2_BAN
				elif readStep >= 12:
					if gameID != "gameid": # if not the first one, add it to the database
						self.X[self.matchCount] = np.copy(line)
						self.y = np.append(self.y, int(row[READ_WINNER]))
					# Reset and start a new line
					readStep = 0
					gameID = row[0]	
					line = [0] * ESPORTS_DATA_LENGTH
					line[DATA_PATCH_NUMBER] = 1000 * int(row[READ_PATCH_NUMBER]) # Read patch number
					# read team 1 data
					line[binarySearch(self.teams, row[READ_TEAM_NAME]) + DATA_TEAM_NAMES] = TEAM_1
					for i in range(5):
						line[binarySearch(self.champions, row[READ_BANS + i]) + DATA_CHAMP_NAMES] = TEAM_1_BAN

				if readStep < 5:
					line[binarySearch(self.champions, row[READ_CHAMPION]) + DATA_CHAMP_NAMES] = TEAM_1_PICK
				elif readStep < 10:
					line[binarySearch(self.champions, row[READ_CHAMPION]) + DATA_CHAMP_NAMES] = TEAM_2_PICK
					
				readStep = readStep + 1

	def partitionData(self, percentage):
		data = ESportsData()
		maximum = int(percentage * self.y.shape[0]) # how much is being used as test data
		testingRows = np.random.choice(self.y.shape[0], maximum, replace=False) # Generate a list of random rows
		data.X = self.XText[testingRows]
		data.y = self.y[testingRows]
		self.X = np.delete(self.XText, testingRows, axis=0)
		self.y = np.delete(self.y, testingRows)
		return data

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