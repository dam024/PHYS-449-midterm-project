import torch
import torch.nn as nn
import Generator as G
import InputManager as IM

class NeuralNetwork(nn.Module):

	#params : NN Structure parameters from param.json
	def __init__(self,params):
		super(NeuralNetwork, self).__init__()
		self.generator = G.Generator(params['num_convs'],params['num_layers'],params['initial_filter_num'])

	#Execute the generator and return it's output
	def forward(self,inp):
		out = self.generator.forward(inp)
		return out

	#Execute the critics and return the loss. It takes as arguments the generated halo counts and the real halo counts. 
	def critic(self,generated, real):
		pass

	def trainNetwork(self,inputManager):
		#Define the optimizer

		#Define loss function

		#Train the data. Read in the article how it is done and add the necessary parameters
		pass

if __name__ == '__main__':
	import sys
	sys.path.append('../')
	import main
	main.main('../')