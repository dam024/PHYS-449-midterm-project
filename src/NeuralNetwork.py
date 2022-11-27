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
				return torch.device("cuba:0")
			except:
				warnings.warn("Impossible to find mps device. So the device is set to the cpu.")
				return torch.device("cpu")

	#params : NN Structure parameters from param.json
	def __init__(self,params, trainingParams, isTraining,modelSavingPath,resumeTraining,lossPath):
		super(NeuralNetwork, self).__init__()
		self.lossPath = lossPath
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
	def forwardCritic(self,generated, real):
		realLoss=self.critic.forward(real).mean()
		genLoss=self.critic.forward(generated).mean()
		loss = genLoss-realLoss
		return loss

	def trainNetwork(self,inputManager,params,modelSavingPath):
		print("Start training...\n")
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

		#Train the data. Read in the article how it is done and add the necessary parameters
		for epoch in range(self.epoch,params['epoch']):
			print("Training procedure [{}/{}]".format(epoch+1,params['epoch'])+" :")
			self.epoch = epoch
			data = inputManager.getTrainData(NeuralNetwork.device())

			tmpTrainLoss = []

			print("Training the critic :")
			generated = self.forward(data.x)
			for epochCritic in range(self.epochCritic,params['epoch_critic']):
				self.epochCritic = epochCritic

				train_val = self.forwardCritic(generated, data.y)
				self.critic.backprop(train_val,self.optimizerCritic)
				tmpTrainLoss.append(train_val)

				if (epochCritic+1) % params['display_epochs_critic'] == 0:
					print('Epoch [{}/{}]'.format(epochCritic+1, params['epoch_critic'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
			#Append the new loss result
			print(self.obj_vals)
			print(type(self.obj_vals))
			print(self.obj_vals.keys())
			print(self.obj_vals['critic'])
			self.obj_vals['critic'].append(tmpTrainLoss)
			tmpTrainLoss = []
			print()
			print("Training the generator")
			for epochGenerator in range(self.epochGenerator, params['epoch_generator']):
				self.epochGenerator = epochGenerator

				generated = self.forward(data.x)
				train_val = self.generator.backprop(data.x, generated, self.critic, self.optimizerGenerator)
				tmpTrainLoss.append(train_val)

				if (epochCritic+1) % params['display_epochs_generator'] == 0:
					print('Epoch [{}/{}]'.format(epochGenerator+1, params['epoch_generator'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
			
			#Append loss
			#print(len(self.obj_vals['generator']))
			self.obj_vals['generator'].append(tmpTrainLoss)
			#print(len(self.obj_vals['generator']))
			print()
			print()
			self.epochGenerator = 0
			self.epochCritic = 0
			if (epoch+1) % params['display_epochs'] == 0:
				print('Epoch [{}/{}]'.format(epoch+1, params['epoch'])+\
                      '\tTraining Loss: {:.4f}'.format(train_val))
		self.epoch = params['epoch']#To stop at the correct epoch in case of success
		print("End of training \n")
		if len(self.obj_vals['critic']) > 0 and len(self.obj_vals['generator']) > 0:
			print('Final training results : \tGenerator: {:.4f} \tCritic: {:.4f}'.format(self.obj_vals['generator'][-1][-1],self.obj_vals['critic'][-1][-1]))
		self.saveParameters(modelSavingPath,self.lossPath)
		print("Model saved in file : "+modelSavingPath)
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
			print("Error : File at "+path+" should have extension .pt But don't worry, I thought to that case and I created at this path : "+newPath)
			path = newPath
			
		#FI.writeNumPyArrayIntoFile(self.obj_vals, lossPath)
		#print("Save loss : ",self.obj_vals)
		self.saveOutput(self.obj_vals, lossPath+'.pt')
		print('Loss values saved in file '+lossPath+'.pt')

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
				print("Error : Model File "+path+" does not exist.")
				exit(1)
			else:
				return
		state_dict = torch.load(path)
		self.generator.load_state_dict(state_dict[self.__generatorModelKey])
		self.critic.load_state_dict(state_dict[self.__criticModelKey])

		self.optimizerGenerator.load_state_dict(state_dict[self.__generatorOptimizerKey])
		self.optimizerCritic.load_state_dict(state_dict[self.__criticOptimizerKey])
		#print("Resume loss : ",
		self.obj_vals = self.resumeLoss(lossPath+'.pt')
		print("Loss from "+lossPath+'.pt has been resumed')
		#print(self.obj_vals)
		if resumeTraining:
			self.epoch = state_dict[self.__epochKey]
			self.epochCritic = state_dict[self.__epochKeyCritic]
			self.epochGenerator = state_dict[self.__epochKeyGenerator]
			print("Training has been resumed")

		self.generator.eval()
		self.critic.eval()

	def saveOutput(self,output,fileName):
		torch.save(output, fileName)

	#Resume loss
	def resumeLoss(self,fileName):
		#arr = FI.readNumPyArray(fileName)
		arr = []
		if os.path.isfile(fileName):
			#print(fileName)
			arr = torch.load(fileName)
			#print(arr)
		else:
			print("File "+fileName+" does not exists.")
		if arr != []:
			self.obj_vals = arr
		else:
			self.obj_vals = NeuralNetwork.initLossArray()
		return self.obj_vals

	def initLossArray():
		return {'generator':[],'critic':[]}

if __name__ == '__main__':
	import sys
	sys.path.append('../')
	import main
	main.main('../')
