import json
import random
import os
import threading

class neuron():
    def __init__(self):
        userInput = int(input("Whire 1 to create an ann, 0 to load: "))
        if userInput:
            self.numberOfInputs = int(input("Write an numberOfInputs: "))
            self.numberOfHiddenLayers = int(input("Write an numberOfHiddenLayers: "))
            self.numberOfOutputs = int(input("Write an numberOfOutputs: "))
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
        else:
            name = input("Write a name of your ANN: ")
            if os.path.exists(name):
                self.name = name
                self.loadDataJson()
            else:
                print("ANN does not exiat")

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
        self.numberOfOutputs = data["numberOfOutputs"]
        self.numberOfHiddenLayers = data["numberOfHiddenLayers"]
        self.turnedOnNeurons = data["turnedOnNeurons"]
        self.name = data["name"]

    def sigmoidCounter(self, rawNeuronData):
        e = 2.718281828
        sigmoid = [(1 / (1 + e ** (-x))) for x in rawNeuronData]
        return sigmoid

    def matrixMultiThreadMultiply(self, layerTnput, layerWeights, layerBias):
        numThreads = 4

        result = layerBias

        def multiplyMatrices(start, end):
            for i in range(start, end):
                for j in range(len(layerWeights[0])):
                    result[i] += layerWeights[i][j] * layerTnput[j]

        thread_list = []
        for i in range(numThreads):
            start = i * len(layerWeights) // numThreads
            end = (i + 1) * len(layerWeights) // numThreads
            thread = threading.Thread(target=multiplyMatrices, args=(start, end))
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            thread.join()

        return result

    def calculateNeuronNetwork(self, input):
        layerInput = input
        for layer in range(self.numberOfHiddenLayers + 1):
            layerWeights = self.weights[layer]
            layerBias = self.bias[layer]
            layerInput = self.sigmoidCounter(self.matrixMultiThreadMultiply(layerInput, layerWeights, layerBias))
        return layerInput

    def tester(self, correctInputOutput):
        passes = 0
        for x in correctInputOutput:
            print("Input:", x[0])
            rawOutput = self.calculateNeuronNetwork(x[0])
            print("Raw output: ", rawOutput)
            print("Correct output: ", x[1])
            roundedOutput = [round(i) for i in rawOutput]
            print("Rounded output: ", roundedOutput)
            if roundedOutput == x[1]:
                passes += 1
        print(f"===============\n{passes}/{len(correctInputOutput)}\n===============")

    def teacher(self, correctInputOutput, teachingSessions):
        def countDelta(output1, output2):
            result = [abs(output1[x] - output2[x]) for x in range(len(output1))]
            return result

        def isResultBetter(originalOutput, twikedOutput, correctOutput):
            deltaOriginal = countDelta(correctOutput, originalOutput)
            deltaTwiked = countDelta(correctOutput, twikedOutput)

            compoundDeltaOriginal = sum(num ** 2 for num in deltaOriginal)
            compoundDeltaTwiked = sum(num ** 2 for num in deltaTwiked)

            if compoundDeltaOriginal < compoundDeltaTwiked:
                return 1
            return 0

        for i in range(1, teachingSessions+1):

            testMaterial = random.choice(correctInputOutput)
            inputLesson = testMaterial[0]
            correctOutput = testMaterial[1]

            randomLayer = random.randint(0, self.numberOfHiddenLayers)
            randomNeuron = random.randint(0, self.numberOfInputs-1)
            randomWeight= random.randint(0, self.numberOfInputs-1)
            twik = random.uniform(-0.1, 0.1)
            if self.turnedOnNeurons[randomLayer][randomNeuron]:
                originalOutput = self.calculateNeuronNetwork(inputLesson)  # use original ann
                if random.randint(0,1):
                    # do weights

                    originalWeight = self.weights[randomLayer][randomNeuron][randomWeight]
                    self.weights[randomLayer][randomNeuron][randomWeight] += twik
                    twikedOutput = self.calculateNeuronNetwork(inputLesson) # use new ann

                    if isResultBetter(originalOutput, twikedOutput, correctOutput):
                        self.weights[randomLayer][randomNeuron][randomWeight] = originalWeight

                else:
                    # do bias
                    originalBias = self.bias[randomLayer][randomNeuron]
                    self.bias[randomLayer][randomNeuron] += twik
                    twikedOutput = self.calculateNeuronNetwork(inputLesson)

                    if isResultBetter(originalOutput, twikedOutput, correctOutput):
                        self.bias[randomLayer][randomNeuron] = originalBias



            print(f"{i}th session")

        self.storeDataJson()

test = neuron()
correctInputOutput = [[0,0], [0, 0]],\
                     [[0,1], [1, 0]],\
                     [[1,0], [1, 0]],\
                     [[1,1], [0, 1]]
test.teacher(correctInputOutput, 10)
test.tester(correctInputOutput)
