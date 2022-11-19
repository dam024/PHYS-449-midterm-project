import FileInteraction as FI
import torch
import torch.nn as nn

class InputManager():
	""" Data manager. It contains and manage all the data. 

		Properties :  
			- N : size of the cube, i.e. size of each data
			- test_input : input test data
			- test_output : expected output test data
			- train_input : input train data
			- train_output : expected output train data
	"""

	#inputPath : path to the input data
	#params : data structure parameters from param.json file
	def __init__(self, inputPath,params):
		data = FI.readFileNumbers(inputPath)

		#Treat input data...
		print("Need to format input data. Temporary code")
		self.N = 10 #Size of the cube

		#The data should be formatted in the following way here : 
		#Input is a 5 dimensional tensor : 1st dimension gives the input number, 2nd dimension just contain the rest, 3rd dimension is x values, 4th dimension is y values and 5th dimension is z values. 
		data = torch.randn(101,1,self.N,self.N, self.N)#Temporary data structure

		#Preparing the test data. We take the first n test data :
		n = int(data.size()[0]*params['ratio_test_data'])
		print(n)

		self.train_input = data[0:n]
		self.test_input = data[n:]
		#self.train_output = 
		#self.test_output = 


if __name__ == '__main__':
	import sys
	sys.path.append('../')
	import main
	main.main('../')