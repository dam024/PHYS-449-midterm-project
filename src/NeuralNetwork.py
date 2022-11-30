import os
import torch
import torch.nn as nn
import numpy as np
import Generator as G
import Critic as C
import InputManager as IM
import warnings
import FileInteraction as FI

class NeuralNetwork:

	__generatorModelKey = 'generator_state_dict'
	__criticModelKey = 'critic_state_dict'
	__generatorOptimizerKey = 'optimizerGenerator'
	__criticOptimizerKey = 'optimizerCritic'
	__epochKey = 'epoch'
	__epochKeyCritic = 'epochCritic'
	__epochKeyGenerator='epochGenerator'

	def device():
		try:
			l = b
			return torch.device("mps")
		except:
			try:
				if torch.cuda.is_available():
					return torch.device("cuda")
				else:
					return torch.device("cpu")
			except:
				warnings.warn("Impossible to find mps device. So the device is set to the cpu.")
				return torch.device("cpu")

	#params : NN Structure parameters from param.json
	def __init__(self,params, trainingParams, isTraining,modelSavingPath,resumeTraining,lossPath):
		super(NeuralNetwork, self).__init__()
		self.lossPath = lossPath
		self.verbose = trainingParams['verbose']
		self.generator = G.Generator(params['num_convs'],params['num_layers'],params['initial_filter_num'])
		self.critic = C.Critic()

		#Define the optimizer
		self.optimizerGenerator = torch.optim.Adam(self.generator.parameters(),lr=trainingParams['learning_rate_generator'])
		self.optimizerCritic = torch.optim.Adam(self.critic.parameters(),lr=trainingParams['learning_rate_critic'])
		
		if not isTraining or resumeTraining:
			self.getParameters(modelSavingPath,resumeTraining,lossPath)
		self.generator.to(NeuralNetwork.device())
		self.critic.to(NeuralNetwork.device())

	#Execute the generator and return it's output
	def forward(self,inp):
		out = self.generator.forward(inp)
		return out

	#Execute the critic and return the loss. It takes as arguments the generated halo counts and the real halo counts. 
	def forwardCritic(self, input):
		loss=self.critic.forward(input)
		return loss.mean()

	def trainNetwork(self,inputManager,params,modelSavingPath):
		self.print("Start training...\n")
		#Initialize the epoch parameter if it was not done before. This is to be able to start from previous state if needed
		if not hasattr(self, 'epoch'):
			self.epoch = 0
		if not hasattr(self, 'epochCritic'):
			self.epochCritic = 0
		if not hasattr(self, 'epochGenerator'):
			self.epochGenerator = 0

		#Save loss
		if not hasattr(self, 'obj_vals'):
			self.obj_vals = NeuralNetwork.initLossArray()#{'generator':np.array([]),'critic':np.array([])}

		#one = torch.tensor(1, dtype = torch.float)  #for backproping gradient
		#mone = one*-1                               #for backproping gradient
		#one = one.to(NeuralNetwork.device())
		#one = mone.to(NeuralNetwork.device())

		#Train the data. Read in the article how it is done and add the necessary parameters
		for epoch in range(self.epoch,params['epoch']):
			self.print("Training procedure [{}/{}]".format(epoch+1,params['epoch'])+" :")
			self.epoch = epoch
			data = inputManager.getTrainData(NeuralNetwork.device())

			tmpTrainLoss = []

			##Allow no weight change of generator
			#for p in self.generator.parameters():
			#	p.requires_grad = False
			##allows weight update of critic
			#for p in self.critic.parameters():
			#	p.requires_grad = True

			self.print("Training the critic :")
			self.critic.prepareForBackprop(self.generator)
			generated = self.forward(data.x) 

			for epochCritic in range(self.epochCritic,params['epoch_critic']):
				self.epochCritic = epochCritic
				
				train_val = self.critic.backprop(data,generated,self.forwardCritic,self.optimizerCritic,params)

				tmpTrainLoss.append(train_val)

				if (epochCritic+1) % params['display_epochs_critic'] == 0:
					self.print('Epoch [{}/{}]'.format(epochCritic+1, params['epoch_critic'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
			#Append the new loss result
			self.obj_vals['critic'].append(tmpTrainLoss)
			tmpTrainLoss = []
			
			self.print()
			self.print("Training the generator")

			self.generator.prepareForBackprop(self.critic)
			for epochGenerator in range(self.epochGenerator, params['epoch_generator']):
				self.epochGenerator = epochGenerator

				train_val = self.generator.backprop(data, self.forwardCritic, self.optimizerGenerator,self.critic.mone)

				tmpTrainLoss.append(train_val)

				if (epochCritic+1) % params['display_epochs_generator'] == 0:
					self.print('Epoch [{}/{}]'.format(epochGenerator+1, params['epoch_generator'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
			
			#Append loss
			#self.print(len(self.obj_vals['generator']))
			self.obj_vals['generator'].append(tmpTrainLoss)
			#self.print(len(self.obj_vals['generator']))
			self.print()
			self.print()
			self.epochGenerator = 0
			self.epochCritic = 0
			if (epoch+1) % params['display_epochs'] == 0:
				self.print('Epoch [{}/{}]'.format(epoch+1, params['epoch'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
		self.epoch = params['epoch']#To stop at the correct epoch in case of success
		self.print("End of training \n")
		if len(self.obj_vals['critic']) > 0 and len(self.obj_vals['generator']) > 0:
			self.print('Final training results : \tGenerator: {:.4f} \tCritic: {:.4f}'.format(self.obj_vals['generator'][-1][-1],self.obj_vals['critic'][-1][-1]))
		self.saveParameters(modelSavingPath,self.lossPath)
		self.print("Model saved in file : "+modelSavingPath)
		return self.obj_vals

	#This method will save all the training parameters, so that we can reuse them in a futur run of the program
	def saveParameters(self,path,lossPath):
		folderPath = FI.getFolderPath(path)
		fileName, extension = FI.getFileName(path)

		#make sure all folders are created
		if not os.path.exists(folderPath):
			os.makedirs(folderPath)
		#make sure the file exists. If not, we create one
		if extension != ".pt":
			newPath = folderPath+'/'+fileName+'.pt'
			self.print("Error : File at "+path+" should have extension .pt But don't worry, I thought to that case and I created at this path : "+newPath)
			path = newPath
			
		#FI.writeNumPyArrayIntoFile(self.obj_vals, lossPath)
		#self.print("Save loss : ",self.obj_vals)
		self.saveOutput(self.obj_vals, lossPath+'.pt')
		self.print('Loss values saved in file '+lossPath+'.pt')

		torch.save({
			self.__generatorModelKey: self.generator.state_dict(),
			self.__criticModelKey: self.critic.state_dict(),
			self.__generatorOptimizerKey: self.optimizerGenerator.state_dict(),
			self.__criticOptimizerKey: self.optimizerCritic.state_dict(),
			self.__epochKey: self.epoch,
			self.__epochKeyCritic: self.epochCritic,
			self.__epochKeyGenerator: self.epochGenerator
			},path)


	#This method will get all the training parameters we saved last time. 
	def getParameters(self,path,resumeTraining,lossPath):
		if not os.path.exists(path):
			if not resumeTraining:
				self.print("Error : Model File "+path+" does not exist.")
				exit(1)
			else:
				self.print("No model to retrive...")
				return
		state_dict = torch.load(path)
		self.generator.load_state_dict(state_dict[self.__generatorModelKey])
		self.critic.load_state_dict(state_dict[self.__criticModelKey])

		self.optimizerGenerator.load_state_dict(state_dict[self.__generatorOptimizerKey])
		self.optimizerCritic.load_state_dict(state_dict[self.__criticOptimizerKey])
		#self.print("Resume loss : ",
		self.obj_vals = self.resumeLoss(lossPath+'.pt')
		if self.obj_vals != NeuralNetwork.initLossArray():
			self.print("Loss from "+lossPath+'.pt has been resumed')
		#self.print(self.obj_vals)
		if resumeTraining:
			self.epoch = state_dict[self.__epochKey]
			self.epochCritic = state_dict[self.__epochKeyCritic]
			self.epochGenerator = state_dict[self.__epochKeyGenerator]
			self.print("Training has been resumed")

		self.generator.eval()
		self.critic.eval()

	def saveOutput(self,output,fileName):
		torch.save(output, fileName)

	#Resume loss
	def resumeLoss(self,fileName):
		#arr = FI.readNumPyArray(fileName)
		arr = []
		if os.path.isfile(fileName):
			#self.print(fileName)
			arr = torch.load(fileName)
			#self.print(arr)
		else:
			self.print("File "+fileName+" does not exists.")
		if arr != []:
			self.obj_vals = arr
		else:
			self.obj_vals = NeuralNetwork.initLossArray()
		return self.obj_vals

	def initLossArray():
		return {'generator':[],'critic':[]}

	def print(self,*args):
		if self.verbose == 1:
			print(*args)


if __name__ == '__main__':
	import sys
	sys.path.append('../')
	import main
	main.main('../')
