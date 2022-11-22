import os
import sys
import json
import numpy as np


def readFileNumbers(fileName):
    array = [];
    if os.path.isfile(fileName):
        with open(fileName) as file:
            lines = file.readlines()
            for i in range(len(lines)):
                array.append([float (val) for val in lines[i].split()])
    else:
        print("File "+fileName+" does not exists.")
        exit()
    return array;

def readFileJson(fileName):
    data = {}
    if os.path.isfile(fileName):
        f = open(fileName)
        data = json.load(f)
        f.close()
    else:
        print("File "+fileName+" does not exists.")
        exit()
    return data

def writeArrayIntoFile(array,fileName,mode='w+',dFormat='0.4f'):
    folder = getFolderPath(fileName)
    if not os.path.exists(folder):
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(folder)
    with open(fileName,mode) as file:
        for i in range(len(array)):
            if isinstance(array[i],list):
                for j in range(len(array[i])):
                    file.write(format(array[i][j],dFormat) + ' ')
                file.write('\n')
            else:
                file.write(format(array[i],dFormat) + '\n')

#Save a NumPy array into a file. The output file is the encoded
def writeNumPyArrayIntoFile(array, filename):
    folder = getFolderPath(filename)
    if not os.path.exists(folder):
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(folder)
    np.save(filename, array)
#Load a NumPy array saved into a file
def readNumPyArray(fileName):
    if os.path.isfile(fileName):
        return np.load(fileName)
    else:
        print("File "+fileName+" does not exists.")
        return np.array([])

#Get filename without extension
def getFileName(path):
    path, extension = os.path.splitext(path)
    splited = path.split('/')
    return splited[-1], extension
def getFolderPath(path):
    path, extension = os.path.splitext(path)
    splited = path.split('/')
    return '/'.join(splited[0:-1])


if __name__ == "__main__":
    pass
                