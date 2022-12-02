import  torch
import os
import sys
import argparse
sys.path.append('src')

import FileInteraction as FI
import InputManager as IM
import NeuralNetwork as NN

def testExport(args):
	lossPath = 'results/loss'
	epsilon = 1e-5
	param = FI.readFileJson('parameters/param_test.json')

	inputManager = IM.InputManager('input/training',param['DataStructure'])
	dataTest = inputManager.getBox(0, 0, 0, 0)

	network1 = NN.NeuralNetwork(param['NN_structure'],param['training'], isTraining=True, modelSavingPath='model/model_test.pt',resumeTraining=False,lossPath=lossPath)

	#We train the network
	network1.trainNetwork(inputManager,param['training'],'model/model_test.pt')
	lossValues1 = network1.resumeLoss(lossPath)

	if not os.path.exists('results/loss_generator.txt') or not os.path.exists('results/loss_critic.txt'):
		assert(False),"Loss has not been saved"
	if not os.path.exists('model/model_test.pt'):
		assert(False),"Model has not been saved"
	print("Creation of exported model success")
	### The netwrok must be saved now. So we simulate a resume of training
	param['training']['epoch'] += 1
	network2 = NN.NeuralNetwork(param['NN_structure'], param['training'], isTraining=True, modelSavingPath='model/model_test.pt', resumeTraining=True, lossPath=lossPath)
	#Train the new network
	network2.trainNetwork(inputManager, param['training'], modelSavingPath='model/model_test.pt')
	lossValues2 = network2.resumeLoss(lossPath)

	#Test of the loss
	assert(len(lossValues1['critic']) == len(lossValues2['critic'])-1),"Unexpected size for the resumed network. Got "+str(len(lossValues2['critic']))+" but expected "+str(len(lossValues1['critic'])+1)
	for i in range(len(lossValues1)):
		if lossValues1['critic'][i] != lossValues2['critic'][i]:
			assert(False),"Loss for the Critic is not the same. Got "+str(lossValues1['critic'][i])+" and "+str(lossValues2['critic'][i])
		if lossValues1['generator'][i] != lossValues2['generator'][i]:
			assert(False),"Loss for the Generator is not the same"
	print("Resume of training success")
	### Test without resuming the training
	network3 = NN.NeuralNetwork(param['NN_structure'], param['training'], isTraining=True, modelSavingPath='model/model_test.pt', resumeTraining=False, lossPath=lossPath)
	#Train the new network
	network3.trainNetwork(inputManager, param['training'], modelSavingPath='model/model_test.pt')
	lossValues3 = network3.resumeLoss(lossPath)
	#Check that the two trainings are different
	assert(lossValues3 != lossValues2),"Two different training have the exact same loss"
	
	output3 = network3.forward(dataTest.x)
	loss3 = network3.forwardCritic(output3)#, dataTest.y)

	print("Training without resuming success")
	## Check that the deployment procedure is working
	network4 = NN.NeuralNetwork(param['NN_structure'], param['training'], isTraining=False, modelSavingPath='model/model_test.pt', resumeTraining=False, lossPath=lossPath)
	lossValues4 = network4.resumeLoss(lossPath)
	output4 = network4.forward(dataTest.x)
	loss4 = network4.forwardCritic(output4)#, dataTest.y)
	assert(((output3 - output4) > epsilon).int().sum().item() == 0),"Unexcepted output for the generator. We expected they would be the same, but they aren't. Got : network4 "+str(output4)+" network3 "+str(output3)
	assert(((loss3 - loss4) > epsilon).int().sum().item() == 0),"Unexcepted output for the critic. We expected they would be the same, but they aren't. Got : network4 "+str(loss4)+" network3 "+str(loss3)
	assert(lossValues4 == lossValues3),"Unexcepted loss retrieved during deployment. We expected to have the same loss than previously. Got : network4 : "+str(lossValues4)+" network3 "+str(lossValues3)
	print("Deployment success ")
	
	print("Test export success")

if __name__ == '__main__':

	parser  = argparse.ArgumentParser(description="""Neural network to paint halos from cosmic density fields of dark matter
		""",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p','--param',help='Path to json file containing the parameters for the program. See example at default location.',default='parameters/param_used.json')
	parser.add_argument('-r','--result',help='Path to a folder where the results will be created. Each trials should have its own folder, so that no data get lost !',default="results")
	parser.add_argument('-i','--input',help='Path to the input data -> need to be specified. The path focus on the folder where two files are stored : input.npy and expected.npy. The first one contains the data of the dark matter density field and the second one the data for the halo count density field.',default='input/training')
	parser.add_argument('-t',help='Indicate that we should train our model', action='store_true', dest='isTraining')
	parser.add_argument('-m','--model',help='Path to a folder containing the model/where the model will be stored. If the flag -t is specified, the model will be trained and save the model when done in this file, even if a previous model was saved here. If the -t flag is not specified, it will just load the data from the model.',default="model/model.pt")
	parser.add_argument('-rt','--resume_training',help="If this parameter is specified, training will be resumed at the latest saved state given by the file from the -m parameter. If the file do not exist, this parameter is ignored. Do not use this flag without the -t flag !", action='store_true',dest='resumeTraining')
	args = parser.parse_args()

	args.resumeTraining = args.resumeTraining and args.isTraining #So that we are sure that there is no problem with the resumeTraining parameter in case we are not training. 
	print("Run on device : ",NN.NeuralNetwork.device())
	testExport(args)
	exit()
	out1 = torch.load('results/train.txt')
	out2 = torch.load('results/test.txt')

	epsilon = 1e-5
	print((out1 - out2) < epsilon)
	print(out1)
	print(out2)

	res = ((out1 - out2) > epsilon).float().sum().item()
	if res == 0:
		print('Test succeed')
	else:
		print("Test failed")
