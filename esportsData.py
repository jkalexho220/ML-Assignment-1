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

READ_GAME_STATUS = 1
READ_PATCH_NUMBER = 9
READ_WINNER = 24
READ_TEAM_NAME = 15
READ_CHAMPION = 17
READ_BANS = 18
READ_PLAYER_NAME = 13

TEAM_1_PICK = 2
TEAM_1_BAN  = -1
TEAM_2_PICK = -2
TEAM_2_BAN  = 1

TEAM_1 = 1
TEAM_2 = -1

npStr = np.dtype(('U', 32))

def readByteString(bs):
	if type(bs) == type("hello"):
		return bs.encode('UTF-8', 'replace')
	else:
		return bs

def binarySearch(listName, name):
	point = len(listName) // 2
	left = 0
	right = len(listName) - 1
	while left != right:
		if name == listName[point]:
			return point
		elif name > listName[point]:
			left = point
		else:
			right = point
		point = (left + right) // 2
		if right - left == 1:
			if listName[right] == name:
				return right
			elif listName[left] == name:
				return left
			else:
				raise Exception("Name not found. Name was: " + readByteString(name).decode("UTF-8", "ignore"))
	return -1

class ESportsData:
	def __init__(self):
		self.y = np.empty([0,], dtype=np.int8)
		self.files = ["League/2022.csv", "League/2021.csv", "League/2020.csv"]
		self.matchCount = 0
		self.teams = set()
		self.champions = set()

	def readData(self):
		global DATA_CHAMP_NAMES
		global DATA_TEAM_NAMES
		for f in self.files:
			self.gatherData(f)
		self.esportsDataLength = len(self.teams) + len(self.champions) + 1
		self.X = np.zeros([1 + self.matchCount // 12, self.esportsDataLength], dtype=np.int8)
		self.matchCount = 0
		self.teams = list(self.teams)
		self.champions = list(self.champions)
		self.teams.sort()
		self.champions.sort()
		del self.champions[0]
		DATA_CHAMP_NAMES = DATA_TEAM_NAMES + len(self.teams)
		for f in self.files:
			self.readDataFile(f)
		# validate that data is working
		for i in range(len(self.teams)):
			if self.X[0,DATA_TEAM_NAMES + i] != 0:
				print(self.teams[i])
		for i in range(len(self.champions)):
			if self.X[0,DATA_CHAMP_NAMES + i] != 0:
				print(self.champions[i] + str(self.X[0,DATA_CHAMP_NAMES + i]))

	def gatherData(self, filename):
		with open(filename, newline='', encoding='UTF-8') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			partials = 1
			for row in reader:
				if row[READ_GAME_STATUS] == 'complete':
					self.teams.add(readByteString(row[READ_TEAM_NAME]))
					self.champions.add(row[READ_CHAMPION])
				else:
					partials = partials + 1
			self.matchCount = self.matchCount + reader.line_num - partials

	def readDataFile(self, filename):
		readStep = 0
		with open(filename, newline='', encoding='UTF-8') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			readStep = 12
			line = np.zeros([self.esportsDataLength,], dtype=np.int8)
			for row in reader:
				if row[READ_GAME_STATUS] == 'complete':
					if readStep == 11:
						# read team 2 data
						try:
							line[binarySearch(self.teams, readByteString(row[READ_TEAM_NAME])) + DATA_TEAM_NAMES] = TEAM_2
						except Exception as e:
							print(reader.line_num)
							print(filename)
							raise

						for i in range(5):
							if len(row[READ_BANS + i]) > 0:
								line[binarySearch(self.champions, row[READ_BANS + i]) + DATA_CHAMP_NAMES] = TEAM_2_BAN

						self.X[self.matchCount] = np.copy(line)
						self.y = np.append(self.y, int(row[READ_WINNER]))
						self.matchCount = self.matchCount + 1
					elif readStep >= 12:
						# Reset and start a new line
						readStep = 0
						line = np.zeros([self.esportsDataLength,], dtype=np.int8)
						try:
							line[DATA_PATCH_NUMBER] = int(float(row[READ_PATCH_NUMBER])*100) # Read patch number
						except Exception as e:
							print(filename)
							print(reader.line_num)
							raise
						# read team 1 data
						line[binarySearch(self.teams, readByteString(row[READ_TEAM_NAME])) + DATA_TEAM_NAMES] = TEAM_1
						for i in range(5):
							if len(row[READ_BANS + i]) > 0:
								line[binarySearch(self.champions, row[READ_BANS + i]) + DATA_CHAMP_NAMES] = TEAM_1_BAN

					if readStep < 5:
						line[binarySearch(self.champions, row[READ_CHAMPION]) + DATA_CHAMP_NAMES] = TEAM_1_PICK
					elif readStep < 10:
						line[binarySearch(self.champions, row[READ_CHAMPION]) + DATA_CHAMP_NAMES] = TEAM_2_PICK
						
					readStep = readStep + 1

	def partitionData(self, percentage):
		data = ESportsData()
		data.esportsDataLength = self.esportsDataLength
		data.matchCount = self.matchCount
		data.teams = self.teams
		data.champions = self.champions
		maximum = int(percentage * self.y.shape[0]) # how much is being used as test data
		testingRows = np.random.choice(self.y.shape[0], maximum, replace=False) # Generate a list of random rows
		data.X = self.X[testingRows]
		data.y = self.y[testingRows]
		self.X = np.delete(self.X, testingRows, axis=0)
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