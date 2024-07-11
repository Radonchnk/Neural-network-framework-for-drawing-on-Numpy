from tools.createDataset import main as dataSetMain
from tools.drawCheck import DrawingApp
from tools.trainNN import train
from tools.Network import Network

def userInterface(NNname, dSet):
    while True:
        userInput = int(input("=====\n0 - manage dataSet\n1 - trainNN\n2 - draw and check\n=====\n"))
        if userInput == 0:
            dataSetMain(dSet)
        elif userInput == 1:
            cycles = int(input("Write number of cycles: "))
            batchSize = int(input("Write batchSize: "))
            eta = int(input("Write eta: "))
            train(DSetName=dSet, NNname=NNname, eta=eta, cycles=cycles, batchSize=batchSize)
        elif userInput == 2:
            DrawingApp(NNname)

def main():
    has_neural_network = input("Do you have a neural network? (yes/no): ").lower() == "yes"

    if has_neural_network:
        NNname = input("Write a name of neural network: ")
    else:
        nn = Network(userInput=1)
        NNname = nn.name

    dSet = input("Write a name of dataset to create or use: ")

    userInterface(NNname, dSet)

if __name__ == "__main__":
    main()