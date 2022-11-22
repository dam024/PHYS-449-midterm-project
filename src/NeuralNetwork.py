import os
import torch
import torch.nn as nn
import Generator as G
import Critic as C
import InputManager as IM
import warnings
import FileInteraction as FI

class NeuralNetwork:

	__generatorModelKey = 'generator_state_dict'
	__criticModelKey = 'critic_state_dict'

	def device():
		try:
			l = b
			return torch.device("mps")
		except:
			try:
				return torch.device("cuba:0")
			except:
				warnings.warn("Impossible to find mps device. So the device is set to the cpu.")
				return torch.device("cpu")

	#params : NN Structure parameters from param.json
	def __init__(self,params, isTraining,modelSavingPath):
		super(NeuralNetwork, self).__init__()
		self.generator = G.Generator(params['num_convs'],params['num_layers'],params['initial_filter_num'])
		warnings.warn('Complete the critic initialization parameters')
		self.critic = C.Critic()
		if not isTraining:
			self.getParameters(modelSavingPath)
		self.generator.to(NeuralNetwork.device())
		self.critic.to(NeuralNetwork.device())

	#Execute the generator and return it's output
	def forward(self,inp):
		out = self.generator.forward(inp)
		return out

	#Execute the critic and return the loss. It takes as arguments the generated halo counts and the real halo counts. 
	def forwardCritic(self,generated, real):
		warnings.warn('Implement here the execution of the critic. It correspond to a forward method.')
		return 0

	def trainNetwork(self,inputManager,params,modelSavingPath):
		print("Start training...\n")
		#Define the optimizer
		warnings.warn('Define the optimizer here. I don\'t know which one it is')
		#Define loss function
		warnings.warn('Define the loss function here. This is the Critic')

		#Save loss
		obj_vals = []

		#Train the data. Read in the article how it is done and add the necessary parameters
		for epoch in range(params['epoch']):
			data = inputManager.getTrainData(NeuralNetwork.device())

			warnings.warn('Training process is maybe not correct... waiting for the critic and the generator to be done.')
			generated = self.forward(data.x)
			train_val = self.forwardCritic(generated, data.y)
			obj_vals.append(train_val)

			warnings.warn('Implement backward propagation here')

			if (epoch+1) % params['display_epochs'] == 0:
				print('Epoch [{}/{}]'.format(epoch+1, params['epoch'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
		print('Final training results : \tloss: {:.4f}'.format(obj_vals[-1]))
		print("End of training \n")
		self.saveParameters(modelSavingPath)
		print("Model saved in file : "+modelSavingPath)
		return obj_vals

	#This method will save all the training parameters, so that we can reuse them in a futur run of the program
	def saveParameters(self,path):
		folderPath = FI.getFolderPath(path)
		fileName, extension = FI.getFileName(path)

		#make sure all folders are created
		if not os.path.exists(folderPath):
			os.makedirs(folderPath)
		#make sure the file exists. If not, we create one
		if extension != ".pt":
			newPath = folderPath+'/'+fileName+'.pt'
			print("Error : File at "+path+" should have extension .pt But don't worry, I thought to that case and I created at this path : "+newPath)
			path = newPath
			
		torch.save({
			self.__generatorModelKey: self.generator.state_dict(),
			self.__criticModelKey: self.critic.state_dict(),
			},path)


	#This method will get all the training parameters we saved last time. 
	def getParameters(self,path):
		if not os.path.exists(path):
			print("Error : Model File "+path+" does not exist.")
			exit(1)
		state_dict = torch.load(path)
		self.generator.load_state_dict(state_dict[self.__generatorModelKey])
		self.critic.load_state_dict(state_dict[self.__criticModelKey])

		self.generator.eval()
		self.critic.eval()

	def saveOutput(self,output,fileName):
		torch.save(output, fileName)

if __name__ == '__main__':
	import sys
	sys.path.append('../')
	import main
	main.main('../')