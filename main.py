#Importing pytorch here is forbidden
import argparse
import sys
import os
import warnings

sys.path.append('src')
import FileInteraction as FI
import InputManager as IM
import NeuralNetwork as NN

def main(prefix):
	parser  = argparse.ArgumentParser(description="""Neural network to paint halos from cosmic density fields of dark matter
		""",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p','--param',help='Path to json file containing the parameters for the program. See example at default location.',default=prefix+'parameters/param.json')
	parser.add_argument('-r','--result',help='Path to a folder where the results will be created. Each trials should have its own folder, so that no data get lost !',default=prefix+"results")
	parser.add_argument('-i','--input',help='Path to the input data -> need to be specified',default=prefix+'input/input.dat')
	parser.add_argument('-t',help='Indicate that we should train our model', action='store_true', dest='isTraining')
	parser.add_argument('-m','--model',help='Path to a folder containing the model/where the model will be stored. If the flag -t is specified, the model will be trained and save the model when done in this file, even if a previous model was saved here. If the -t flag is not specified, it will just load the data from the model.',default=prefix+"model/model.pt")
	parser.add_argument('-rt','--resume_training',help="If this parameter is specified, training will be resumed at the latest saved state given by the file from the -m parameter. If the file do not exist, this parameter is ignored. Do not use this flag without the -t flag !", action='store_true',dest='resumeTraining')
	args = parser.parse_args()

	args.resumeTraining = args.resumeTraining and args.isTraining #So that we are sure that there is no problem with the resumeTraining parameter in case we are not training. 
	#Get parameters
	param = FI.readFileJson(args.param)

	#Prepare input data
	inputManager = IM.InputManager(args.input,param['DataStructure'])

	#Create network
	network  = NN.NeuralNetwork(param['NN_structure'],param['training'], args.isTraining,args.model,args.resumeTraining)

	#To save/retriev the loss
	lossPath = args.result+'/'+'loss'

	#We resume the loss if necessary, like if we are in deployement mode or we resumeTraining
	lossValues = []
	if not args.isTraining or args.resumeTraining:
		lossValues = network.resumeLoss(lossPath+'.npy')


	if args.isTraining:
		try:
			lossValues = network.trainNetwork(inputManager,param['training'],args.model)
		except:
			network.saveParameters(args.model)
			raise
		print(lossValues)
		FI.writeNumPyArrayIntoFile(lossValues, lossPath)
		print('Loss values saved in file '+lossPath)

	#Use the neural network
	warnings.warn("Use the correct input. Just enter three coordinates to obtain the correct box")
	data = inputManager.getBox(0,0,0)
	output1 = network.forward(data.x)

	#Saving the output
	if args.isTraining:
		network.saveOutput(output1,args.result+'/'+'train.txt')
	else:
		network.saveOutput(output1,args.result+'/'+'test.txt')

	warnings.warn("Implement the analysis of the results after here")



if __name__ == '__main__':
	main('')
