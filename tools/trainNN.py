import pickle
import gzip
from . import Network
import random

def printData(array):
    divided_arrays = [array[i:i + 28] for i in range(0, len(array), 28)]
    for x in divided_arrays:
        for y in x:
            if y == 0:
                print("  ", end="")
            else:
                print("██", end="")
        print("")

def move_image_randomly(image_array):
    image_array = [image_array[i:i + 28] for i in range(0, len(image_array), 28)]

    # Generate random displacements for x and y axes
    delta_x = random.randint(-1, 1)
    delta_y = random.randint(-1, 1)

    # Create a new empty 28x28 array to store the moved image
    moved_image = [[0 for _ in range(28)] for _ in range(28)]

    # Move the image within the range of 3 pixels in x and y axes
    for y in range(28):
        for x in range(28):
            new_x = x + delta_x
            new_y = y + delta_y

            # Check if the new x and y coordinates are within the image boundaries
            if 0 <= new_x < 28 and 0 <= new_y < 28:
                moved_image[new_y][new_x] = image_array[y][x]

    newArray = []
    for x in moved_image:
        newArray += x
    return newArray

class train():
    def __init__(self, DSetName, cycles, batchSize, eta, NNname = ""):
        f = gzip.open(f'{DSetName}.pkl.gz', 'rb')
        train = pickle.load(f, encoding="latin1")

        nn = Network.Network(name = NNname)
        nn.SGD(train, cycles, batchSize, eta, train)

if __name__ == "__main__":
    train(DSetName="amogus",NNname="amogus",cycles=20,batchSize=5,eta=1)