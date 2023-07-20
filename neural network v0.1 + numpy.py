import random
import numpy as np
import json
import os
import pickle
import gzip
import time

class Network(object):

    def __init__(self, name = ""):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        if name == "":
            userInput = int(input("Write 1 to create an ann, 0 to load: "))
        else:
            userInput = 0
        if userInput:
            name = input("Write an name for your neural network: ")
            if os.path.exists(name):
                print("name already taken")
            else:
                self.name = name
                os.makedirs(self.name)
                numberOfInputs = int(input("Write an numberOfInputs: "))
                numberOfHiddenLayers = int(input("Write an numberOfHiddenLayers: "))
                numberOfOutputs = int(input("Write an numberOfOutputs: "))

                neuronsPerHidden = []
                for x in range(numberOfHiddenLayers):
                    numberOfNeurons = int(input(f"Write how many neurons do you want to have on hiden layer No.{x + 1}: "))
                    neuronsPerHidden.append(numberOfNeurons)

                self.sizes = [numberOfInputs] + neuronsPerHidden + [numberOfOutputs]

                self.num_layers = len(self.sizes)
                self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
                self.weights = [np.random.randn(y, x)
                                for x, y in zip(self.sizes[:-1], self.sizes[1:])]

                self.storeDataJson()
        else:
            if name == "":
                name = input("Write a name of your ANN: ")
            if os.path.exists(name):
                self.name = name
                self.loadDataJson()
            else:
                print("ANN does not exiat")

    def storeDataJson(self):
        data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "num_layers": self.num_layers,
            "name": self.name,
            "sizes": self.sizes
        }

        with open(f'{self.name}/{self.name}Data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def loadDataJson(self):
        with open(f'{self.name}/{self.name}Data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]
        self.num_layers = data["num_layers"]
        self.name = data["name"]
        self.sizes = data["sizes"]

    def feedforward(self, a):
        a = np.reshape(a, (len(a), 1)) #!!!!! клстиль !!!!! прибрати коли напишу дот функцію
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def testNetwork(self, testingData):
        def round_to_zero_or_one(numpy_array):
            # Round each element to the nearest integer (0 or 1) and convert to 1D Python list
            rounded_list = np.round(numpy_array).astype(int).tolist()
            rounded_list = [x[0] for x in rounded_list]
            return rounded_list

        passes = 0
        for x in testingData:
            rawOutput = self.feedforward(x[0]).tolist()
            roundedOutput = round_to_zero_or_one(rawOutput)
            if roundedOutput == x[1]:
                passes += 1
        return passes

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = 0):
        training_data = [[np.reshape(training_data[x][0], (len(training_data[0][0]), 1)),
                          np.reshape(training_data[x][1], (len(training_data[0][1]), 1))] for x in
                         range(len(training_data))]
        if test_data:
            test_data = [[np.reshape(test_data[x][0], (len(test_data[0][0]), 1)), test_data[x][1]] for x in
                         range(len(test_data))]
        # костиль, який переводить данні в формат np матриці 784x1
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n = len(training_data)
        for j in range(epochs):

            if j % 1 == 0:
                self.storeDataJson()
                start_time = time.time()
                networkResult = self.testNetwork(test_data)
                end_time = time.time()
                print(f"Epoch {j} nn gives result of {networkResult} / {len(test_data)}, which is calculated in {end_time - start_time}s")

            start_time = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data != 0:
                    end_time = time.time()
                    print(f"Epoch {j} complete in {end_time - start_time}s")
            else:
                end_time = time.time()
                print(f"Epoch {j} complete in {end_time - start_time}s")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

f = gzip.open('compressed_data.pkl.gz', 'rb')
train, test ,validation = pickle.load(f, encoding="latin1")

train2, test2 ,validation2 = load_data_wrapper()

nn = Network(name = "")

#print(nn.feedforward(np.reshape(train[0][0], (784, 1))))
nn.SGD(train, 30, 10, 0.1, test)
