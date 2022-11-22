#Importing pytorch here is forbidden
import argparse
import sys
import os

sys.path.append('src')
import FileInteraction as FI
import InputManager as IM
import NeuralNetwork as NN

def main(prefix):
	parser  = argparse.ArgumentParser(description="""Neural network to paint halos from cosmic density fields of dark matter

		A good way 
		""",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p','--param',help='Path to json file containing the parameters for the program. See example at default location.',default=prefix+'parameters/param.json')
	parser.add_argument('-r','--result',help='Path to a folder where the results will be created',default=prefix+"results")
	parser.add_argument('-i','--input',help='Path to the input data',default=prefix+'input/input.dat')
	parser.add_argument('-t',help='Indicate that we should train our data', action='store_true', dest='isTraining')
	parser.add_argument('-m','--model',help='Path to a folder containing the model/where the model will be stored. Leave the structure of the model, i.e. don\'t modify file names or anything inside the folder. If the folder already contains a model and the program is in training mode, the previous model will be overrided',default=prefix+"model/model.pt")
	parser.add_argument('-rs','--resume_training',help="If this parameter is specified, training will be resumed at the latest saved state given by the file from the -m parameter", action='store_true',dest='resumeTraining')
	args = parser.parse_args()

	#Get parameters
	param = FI.readFileJson(args.param)

	#Prepare input data
	inputManager = IM.InputManager(args.input,param['DataStructure'])

	#Create network
	network  = NN.NeuralNetwork(param['NN_structure'], args.isTraining,args.model,args.resumeTraining)

	if args.isTraining:
		try:
			lossValues = network.trainNetwork(inputManager,param['training'],args.model)
		except:
			network.saveParameters(args.model)
			raise
		lossPath = args.result+'/'+'loss'
		FI.writeNumPyArrayIntoFile(lossValues, lossPath)
		print('Loss values saved in file '+lossPath)

	#Use the neural network
	data = inputManager.getBox(0,0,0)
	output1 = network.forward(data.x)

	#Saving the output
	if args.isTraining:
		network.saveOutput(output1,args.result+'/'+'train.txt')
	else:
		network.saveOutput(output1,args.result+'/'+'test.txt')



if __name__ == '__main__':
	main('')
