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
	__epochKey = 'epoch'

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
	def __init__(self,params, isTraining,modelSavingPath,resumeTraining):
		super(NeuralNetwork, self).__init__()
		self.generator = G.Generator(params['num_convs'],params['num_layers'],params['initial_filter_num'])
		warnings.warn('Complete the critic initialization parameters')
		self.critic = C.Critic()
		if not isTraining or resumeTraining:
			self.getParameters(modelSavingPath,resumeTraining)
		self.generator.to(NeuralNetwork.device())
		self.critic.to(NeuralNetwork.device())

	#Execute the generator and return it's output
	def forward(self,inp):
		out = self.generator.forward(inp)
		return out

	#Execute the critic and return the loss. It takes as arguments the generated halo counts and the real halo counts. 
	def forwardCritic(self,generated, real):
		real=self.critic.forward(real)
		gen=self.critic.forward(generated)
		loss = gen-real
		print(loss)
		return loss

	def trainNetwork(self,inputManager,params,modelSavingPath):
		print("Start training...\n")
		#Initialize the epoch parameter if it was not done before. This is to be able to start from previous state if needed
		if not hasattr(self, 'epoch'):
			self.epoch = 0
		#Define the optimizer
		warnings.warn('Define the optimizer here. I don\'t know which one it is')
		optimizerGenerator = torch.optim.Adam(self.generator.parameters(),lr=params['learning_rate_generator'])
		optimizerGenerator = torch.optim.Adam(self.critic.parameters(),lr=params['learning_rate_critic'])
		#Define loss function
		warnings.warn('Define the loss function here. This is the Critic')

		#Save loss
		if not hasattr(self, 'obj_vals'):
			self.obj_vals = []

		#Train the data. Read in the article how it is done and add the necessary parameters
		for epoch in range(self.epoch,params['epoch']):
			self.epoch = epoch
			data = inputManager.getTrainData(NeuralNetwork.device())

			warnings.warn('Training process is maybe not correct... waiting for the critic and the generator to be done.')
			generated = self.forward(data.x)
			train_val = self.forwardCritic(generated, data.y)
			self.obj_vals.append(train_val)

			warnings.warn('Implement backward propagation here')

			if (epoch+1) % params['display_epochs'] == 0:
				print('Epoch [{}/{}]'.format(epoch+1, params['epoch'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
		self.epoch = params['epoch']#To stop at the correct epoch in case of success
		print("End of training \n")
		print('Final training results : \tloss: {:.4f}'.format(self.obj_vals[-1]))
		self.saveParameters(modelSavingPath)
		print("Model saved in file : "+modelSavingPath)
		return self.obj_vals

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
			self.__epochKey: self.epoch
			},path)


	#This method will get all the training parameters we saved last time. 
	def getParameters(self,path,resumeTraining):
		if not os.path.exists(path):
			if not resumeTraining:
				print("Error : Model File "+path+" does not exist.")
				exit(1)
			else:
				return
		state_dict = torch.load(path)
		self.generator.load_state_dict(state_dict[self.__generatorModelKey])
		self.critic.load_state_dict(state_dict[self.__criticModelKey])
		if resumeTraining:
			self.epoch = state_dict[self.__epochKey]
			print("Training has been resumed")

		self.generator.eval()
		self.critic.eval()

	def saveOutput(self,output,fileName):
		torch.save(output, fileName)

	#Resume loss
	def resumeLoss(self,fileName):
		self.obj_vals = FI.readNumPyArray(fileName).tolist()
		return self.obj_vals

if __name__ == '__main__':
	import sys
	sys.path.append('../')
	import main
	main.main('../')
