import sys
sys.path.append('src')

# Importing pytorch here is forbidden
import matplotlib.pyplot as plt
import FileInteraction as FI
import InputManager as IM
import NeuralNetwork as NN
import argparse
import os
import warnings
#import numpy as np


def plot_results(obj_vals, cross_vals, res_path):
    assert len(obj_vals) == len(
        cross_vals), 'Length mismatch between the curves'

    def flattern(array):
        ret = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                ret.append(array[i][j].detach().numpy())
        return ret

    obj_valsFlat = flattern(obj_vals)
    cross_valsFlat = flattern(cross_vals)
    # Plot saved in results folder
    plt.plot(range(len(obj_valsFlat)), obj_valsFlat,
             label="Generator : ", color="blue")
    plt.plot(range(len(cross_valsFlat)), cross_valsFlat,
             label="Critic : ", color="green")
    plt.legend()

    if not os.path.exists(res_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(res_path)
    plt.savefig(res_path+'/loss.pdf')
    plt.close()


def main(prefix):
    parser = argparse.ArgumentParser(description="""Neural network to paint halos from cosmic density fields of dark matter
		""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--param', help='Path to json file containing the parameters for the program. See example at default location.',
                        default=prefix+'parameters/param_used.json')
    parser.add_argument(
        '-r', '--result', help='Path to a folder where the results will be created. Each trials should have its own folder, so that no data get lost !', default=prefix+"results")
    parser.add_argument('-i', '--input', help='Path to the input data -> need to be specified. The path focus on the folder where two files are stored : input.npy and expected.npy. The first one contains the data of the dark matter density field and the second one the data for the halo count density field.', default=prefix+'input/training')
    parser.add_argument('-t', help='Indicate that we should train our model',
                        action='store_true', dest='isTraining')
    parser.add_argument('-m', '--model', help='Path to a folder containing the model/where the model will be stored. If the flag -t is specified, the model will be trained and save the model when done in this file, even if a previous model was saved here. If the -t flag is not specified, it will just load the data from the model.', default=prefix+"model/model.pt")
    parser.add_argument('-rt', '--resume_training', help="If this parameter is specified, training will be resumed at the latest saved state given by the file from the -m parameter. If the file do not exist, this parameter is ignored. Do not use this flag without the -t flag !", action='store_true', dest='resumeTraining')
    args = parser.parse_args()

    # So that we are sure that there is no problem with the resumeTraining parameter in case we are not training.
    args.resumeTraining = args.resumeTraining and args.isTraining

    # To save/retriev the loss
    lossPath = args.result+'/'+'loss'

    # Get parameters
    param = FI.readFileJson(args.param)

    # Prepare input data
    inputManager = IM.InputManager(args.input, param['DataStructure'])

    # Create network
    network = NN.NeuralNetwork(param['NN_structure'], param['training'],
                               args.isTraining, args.model, args.resumeTraining, lossPath)

    # We resume the loss if necessary, like if we are in deployement mode or we resumeTraining
    lossValues = NN.NeuralNetwork.initLossArray()
    # if not args.isTraining or args.resumeTraining:
    #	lossValues = network.resumeLoss(lossPath+'.pt')
    #	print("Loss resumed : ",lossValues)
    #	print(type(lossValues))

    if args.isTraining:
        try:
            if not os.path.exists(args.result):
                os.makedirs(args.result)
            # empty loss files
            open(lossPath + "_generator.txt", 'w').close()
            open(lossPath + "_critic.txt", 'w').close()
            # calculate
            lossValues = network.trainNetwork(
                inputManager, param['training'], args.model)
        except:
            network.saveParameters(args.model, lossPath)
            raise
        #plot_results(lossValues['generator'], lossValues['critic'], args.result)
        # print(lossValues)
        #FI.writeNumPyArrayIntoFile(lossValues, lossPath)

    # Use the neural network
    warnings.warn(
        "Use the correct input. Just enter three coordinates to obtain the correct box")
    data = inputManager.getBox(0, 0, 0, 0)
    output1 = network.forward(data.x)
    # print(output1)

    # Saving the output
    if args.isTraining:
        network.saveOutput(output1, args.result+'/'+'train.txt')
    else:
        network.saveOutput(output1, args.result+'/'+'test.txt')

    #warnings.warn("Implement the analysis of the results after here")


if __name__ == '__main__':
    main('')
