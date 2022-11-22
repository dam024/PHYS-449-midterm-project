import os
import torch
import torch.nn as nn
import Generator as G
import Critic as C
import InputManager as IM
import warnings

class NeuralNetwork:

	__generatorModelFileName = 'generator.pt'
	__criticModelFileName = 'critic.pt'

	def device():
		try:
			l = b
			return torch.device("mps")
		except:
			print("Warning: Impossible to find mps device. So the device is set to the cpu.")
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
		print("Start training...")
		#Define the optimizer
		warnings.warn('Define the optimizer here. I don\'t know which one it is')
		#Define loss function
		warnings.warn('Define the loss function here. This is the Critic')

		#Save loss
		obj_vals = []

		#Train the data. Read in the article how it is done and add the necessary parameters
		for epoch in range(params['epoch']):
			data = inputManager.getTrainData()

			warnings.warn('Training process is maybe not correct.. waiting for the critic and the generator to be done.')
			generated = self.forward(data.x)
			train_val  = self.forwardCritic(generated, data.y)
			obj_vals.append(train_val)
			warnings.warn('Implement backward propagation here')

			if (epoch+1) % params['display_epochs'] == 0:
				print('Epoch [{}/{}]'.format(epoch+1, params['epoch'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
		print('Final training results : \tloss: {:.4f}'.format(obj_vals[-1]))
		self.saveParameters(modelSavingPath)

	#This method will save all the training parameters, so that we can reuse them in a futur run of the program
	def saveParameters(self,path):
		#warnings.warn('Implement the method to save the parameters')
		#Save the generator state
		torch.save(self.generator.state_dict(), path+'/'+self.__generatorModelFileName)
		#Save the critic state
		torch.save(self.critic.state_dict(), path+'/'+self.__criticModelFileName)


	#This method will get all the training parameters we saved last time. 
	def getParameters(self,path):
		#warnings.warn('Implement the method to get the parameters')
		#load the generator
		self.generator = NeuralNetwork.loadModel(self.generator, path+'/'+self.__generatorModelFileName)
		#load the critic
		self.critic = NeuralNetwork.loadModel(self.critic,path+'/'+self.__criticModelFileName)

	def loadModel(model,fileName):
		if not os.path.exists(fileName):
			print("File "+fileName+" does not exist. You shouldn't have touched anything that is the same directory... program stopped")
		model.load_state_dict(torch.load(fileName))
		model.eval()
		#model.train()
		return model

	def saveOutput(self,output,fileName):
		torch.save(output, fileName)

if __name__ == '__main__':
	import sys
	sys.path.append('../')
	import main
	main.main('../')