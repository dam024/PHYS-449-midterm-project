import FileInteraction as FI
import torch
import torch.nn as nn
import random
import numpy as np
import warnings
import os

class Data:
	def __init__(self):
		pass

class InputManager():
	""" Data manager. It contains and manage all the data. 

		Properties :  
			- N : size of the cube, i.e. size of each data
			- dataInput : input test -> do not access directly
			- dataOutput : expected output test data -> do not access directly
		Methods : 
			- getTrainData : get a random sample of the train data
	"""

	#inputPath : path to the input data
	#params : data structure parameters from param.json file
	def __init__(self, inputPath,params):
		seed  = params['random_seed']
		self.size = params['training_size']
		if seed != None:
			random.seed(seed)
		#Read input data
		#data = FI.readFileNumbers(inputPath)

		self.nTrainBox = int(len(os.listdir(inputPath)) / 2)

		for i in range(self.nTrainBox):
			# The data should be formatted in the following way here :
			# Input is a 5 dimensional tensor : 1st dimension gives the input number,
			# 2nd dimension just contain the rest, 3rd dimension is x values,
			# 4th dimension is y values and 5th dimension is z values.
			inpPath = '{}/emulator_720box_planck_00-{}_logmass-12-14_halo_counts.npy'.format(inputPath, i)
			expectedPath = '{}/emulator_720box_planck_00-{}_particle_counts.npy'.format(inputPath, i)
			if os.path.exists(inpPath) and os.path.exists(expectedPath):
				if i == 0:
					firstInput = np.load(inpPath)
					self.N = firstInput.shape[0]
					self.dataInput = np.empty((self.nTrainBox, self.N, self.N, self.N))
					self.dataInput[i, :, :, :] = firstInput
					self.dataOutput[i, :, :, :] = np.load(expectedPath)

				else:
					self.dataInput[i, :, :, :] = np.load(inpPath)
					self.dataOutput[i, :, :, :] = np.load(expectedPath)
			else:
				print("Impossible to find output data in folder {} : {} and {}".format(inputPath, inpPath, expectedPath))
				exit(1)

		# self.N = len(self.dataInput) #Size of the cube
		
		if self.size > self.N:
			print("Error : Total data size is smaller than the configured working box...")
			exit(2)
		#Preparing the test data. We take the first n test data :
		#n = int(len(self.dataInput)*params['ratio_test_data'])
		#print(n)

		#self.train_input = data[0:n]
		#self.test_input = data[n:]
		#self.train_output = 
		#self.test_output = 
	#Return a random box from the main data
	def getTrainData(self, device=torch.device("cpu")):
		n = random.randint(0, self.nTrainBox - 1)
		xs = random.randint(0, self.N - self.size - 1)
		ys = random.randint(0, self.N - self.size - 1)
		zs = random.randint(0, self.N - self.size - 1)
		xe = xs + self.size
		ye = ys + self.size
		ze = zs + self.size

		#print(xs, xe, ys, ye, zs, ze)
		#print(self.dataInput[xs:xe].shape, self.dataInput[xs:xe][ys:ye].shape, self.dataInput[xs:xe][ys:ye][zs:ze].shape)
		#print(self.dataInput[xs:xe].shape, self.dataInput[xs:xe,ys:ye].shape, self.dataInput[xs:xe,ys:ye,zs:ze].shape)
		data = Data()
		data.x = torch.from_numpy(np.array([[self.dataInput[n, xs:xe, ys:ye, zs:ze]]], dtype=np.float32)).to(device)
		data.y = torch.from_numpy(np.array([[self.dataOutput[n, xs:(xe-8), ys:(ye-8), zs:(ze-8)]]], dtype=np.float32)).to(device)
		#print(data.x.size(), data.y.size(), self.dataInput.shape, self.dataOutput.shape)
		#exit()
		return data
	#Return a defined box from the origin point
	def getBox(self, n, xs, ys, zs, device=torch.device("cpu")):

		xe = xs + self.size
		ye = ys + self.size
		ze = zs + self.size

		data = Data()
		data.x = torch.from_numpy(np.array([[self.dataInput[n, xs:xe, ys:ye, zs:ze]]], dtype=np.float32)).to(device)
		data.y = torch.from_numpy(np.array([[self.dataOutput[n, xs:(xe-8), ys:(ye-8), zs:(ze-8)]]], dtype=np.float32)).to(device)
		#print(data.x.size(),data.y.size())
		#exit()
		return data


if __name__ == '__main__':
	import sys
	sys.path.append('../')
	import main
	main.main('../')