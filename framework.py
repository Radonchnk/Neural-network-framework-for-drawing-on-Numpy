import json
import random
import os

class neuron():
    def __init__(self, numberOfInputs, numberOfHiddenLayers, numberOfOutputs):
        self.numberOfInputs = numberOfInputs
        self.numberOfHiddenLayers = numberOfHiddenLayers
        self.numberOfOutputs = numberOfOutputs
        name = input("Write an name for your neural network: ")
        if os.path.exists(name):
            print("name already taken")
        else:
            self.name = name
            os.makedirs(self.name)

            weightsIndividual = [[0] * self.numberOfInputs]
            weightsLayer = [weightsIndividual * self.numberOfInputs]
            self.weights = weightsLayer * (self.numberOfHiddenLayers+1)

            biasLayer = [[0] * self.numberOfInputs]
            self.bias = biasLayer * (self.numberOfHiddenLayers+1)


            self.turnedOnNeurons = [[1] * self.numberOfInputs] * (self.numberOfHiddenLayers) +\
                              [([1] * self.numberOfOutputs + [0] * (self.numberOfInputs - self.numberOfOutputs))]


            self.storeDataJson()

            self.loadDataJson()

    def storeDataJson(self):
        data = {
            "weights": self.weights,
            "bias": self.bias,
            "numberOfInputs": self.numberOfInputs,
            "numberOfHiddenLayers": self.numberOfHiddenLayers,
            "numberOfOutputs": self.numberOfOutputs,
            "turnedOnNeurons": self.turnedOnNeurons,
            "name": self.name
        }

        with open(f'{self.name}/{self.name}Data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def loadDataJson(self):
        with open(f'{self.name}/{self.name}Data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.numberOfInputs = data["numberOfInputs"]
        self.numberOfHiddenLayers = data["numberOfHiddenLayers"]
        self.name = data["name"]

neuron(2,1,1)