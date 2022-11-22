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
