# Painting halos from cosmic density fields of dark matter with a neural network

## Description
The objective of this work is to reproduce the results of the paper https://arxiv.org/pdf/1903.10524.pdf. As a short summary, we have tried to implement a neural network, whose goal is to, providing an input dark matter mass density field, generate an halo count density field. 

## Running
Basically, the program can be run using the simple command. This command will use the neural network on the command and save the output. Note that no parameters are required to run the network in deployment mode. 
```sh
python main.py
```

A lot of parameters are described from the help page :
```sh
python main.py -h
```

A basic command for running is to use the -t flag. To resume the training at it's last state, use the flag -rt. The flag -rt may be used only if an error occured during the execution of the program or if the model needs more training, as it will restore the exact same state the model was when it was saved. 
```sh
python main.py -t [-rt]
```
It may be interesting to save the model at each tries, as it is really long to train. Thus, don't forget to use the parameter -m when training. The same parameter is used, with the path to the desired model, to resume the model state in deployement mode :
```sh
python main.py -t -m model/model.pt
```

## Other infos

input :  input data
model : saved models
parameters : hyperparameters + training parameters
results : output graphs/output images/output data/etc
src : all other Python files

compile : it is for my (Damien) setup, because I am lazzy to change 20 times the setup of my code editor


I  created some branches for better management of the work. Please commit on main only working code.  

# Input

## emulator_1100box_planck_products

*Mike: I was originally going to upload the raw halo catalogues here as well as the counts, but the halo files seem to be too large for the upload. I will process the raw halo files and upload only the grid containing the halo counts used by the critic.*

Contains the halo counts from select emulator boxes of the AbacusCosmos simulation suite (https://lgarrison.github.io/AbacusCosmos/simulations/). AbacusCosmos is used to replace the VELMASS simulations because they are publicly available and have code for generating the 2LPT density field. The emulator boxes are chosen since they have boxes that vary along $\Omega_m$ (and related parameters) specifically, allowing reproduction of the varying cosmology test of Ramanah et al. 2020. The z=0.3 slice is used since it is the lowest redshift slice, while Ramanah et al. 2020 used z=0.

Box 00 has the default Planck cosmology and should be used for training, while Box 03 and 04 have higher and lower $\Omega_m$ respectively, and should be used for testing on a variable cosmology.
