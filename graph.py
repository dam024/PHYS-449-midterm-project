import sys
sys.path.append('src')

# Importing pytorch here is forbidden
import matplotlib.pyplot as plt
import FileInteraction as FI
import InputManager as IM
import NeuralNetwork as NN
import argparse
import os
import time

def plot_results(obj_vals, res_path,dtype,length,real_vals=[]):

    def flattern(array):
        ret = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                ret.append(array[i][j])#.detach().numpy())
        return ret

    obj_valsFlat = flattern(obj_vals)
    real_valsFlat = flattern(real_vals)
    # Plot saved in results folder
    plt.plot(range(int(len(obj_valsFlat))), obj_valsFlat,
             label=dtype+" : ", color="blue")
    plt.plot(range(int(len(real_valsFlat))), real_valsFlat,
             label=dtype+" : ", color="red")

    plt.legend()

    if not os.path.exists(res_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(res_path)
    plt.savefig(res_path+'/loss_'+dtype+'.pdf')
    plt.close()

if __name__ == '__main__':
	prefix = ''
	parser = argparse.ArgumentParser(description="""Generate the graphs
		""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-g', help='Graph loss for the generator',action='store_true')
	parser.add_argument('-c', help='Graph loss for the critic',action='store_true')
	parser.add_argument('-i','--input', help='Path to the result folder where the loss are',default='result_test')
	parser.add_argument('-p', '--param', help='Path to json file containing the parameters for the program. See example at default location.',
                        default=prefix+'parameters/param_used.json')
	parser.add_argument('-m', '--model', help='Path to a folder containing the model/where the model will be stored. If the flag -t is specified, the model will be trained and save the model when done in this file, even if a previous model was saved here. If the -t flag is not specified, it will just load the data from the model.', default=prefix+"model/model_test.pt")

	args = parser.parse_args()

	# To save/retriev the loss
	lossPath = args.input+'/'+'loss'

	# Get parameters
	param = FI.readFileJson(args.param)

	network = NN.NeuralNetwork(param['NN_structure'], param['training'],
                               isTraining=False, modelSavingPath=args.model, resumeTraining=False, lossPath=lossPath)

	while True:
		lossVal = network.resumeLoss(lossPath)

		if args.c:
			plot_results(lossVal['critic'], args.input, 'Critic',param['training']['epoch_critic'])
		if args.g:
			plot_results(lossVal['generator'], args.input, 'Generator',param['training']['epoch_generator'],real_vals=lossVal['real'])
		time.sleep(60)