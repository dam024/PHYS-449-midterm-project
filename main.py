#Importing pytorch here is forbidden
import argparse
import sys
import os

sys.path.append('src')
import FileInteraction as FI
import InputManager as IM
import NeuralNetwork as NN

def main(prefix):
	parser  = argparse.ArgumentParser(description='Neural network to paint halos from cosmic density fields of dark matter',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p','--param',help='Path to json file containing the parameters for the program. See example at default location.',default=prefix+'parameters/param.json')
	parser.add_argument('-r','--result',help='Path to a folder where the results will be created',default=prefix+"results")
	parser.add_argument('-i','--input',help='Path to the input data',default=prefix+'input/input.dat')
	parser.add_argument('-t',help='Indicate that we should train our data')
	args = parser.parse_args()

	#Get parameters
	param = FI.readFileJson(args.param)

	#Prepare input data
	inputManager = IM.InputManager(args.input,param['DataStructure'])

	#Create network
	network  = NN.NeuralNetwork(param['NN_structure'])



if __name__ == '__main__':
	main('')
