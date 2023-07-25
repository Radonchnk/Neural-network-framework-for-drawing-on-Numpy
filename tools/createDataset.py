import tkinter as tk
import os
import gzip
import pickle
import io
import random

class main():
    def __init__(self, name = ""):
        if name == "":
            name = input("Write name of new dataset: ")
        if not os.path.exists(f"{name}.pkl.gz"):
            data = []
            storeData(data, name)
        data = loadData(name)

        userInput = int(input("1 to add items to dataset, 0 to add random data to dataset, 2 to print data: "))
        if userInput == 1:
            app = DrawingApp(name)
        elif userInput == 0:
            numberOfRandom = int(input("Write number of random inputs: "))

            for i in range(numberOfRandom):
                array = [random.random() for _ in range(84)]
                array = array + [0] * 700
                random.shuffle(array)
                addToDataset(array, name, 0)
        else:
            data = loadData(name)
            for x in data:
                print(x[0])
                print(x[1])
            print(len(data))

class DrawingApp:
    def __init__(self, name):
        # Initialize the canvas dimensions and pixel size
        self.canvas_width = 280
        self.canvas_height = 280
        self.pixel_size = 1
        self.SetName = name
        self.sector_size = 10  # Size of each sector

        # Initialize the 280x280 grid as a list of lists (all set to 0 initially)
        self.grid = [[0 for _ in range(self.canvas_width)] for _ in range(self.canvas_height)]

        # Initialize the values for the panel on the left side of the grid
        self.panel_values = [0] * 10

        # Create the GUI
        self.root = tk.Tk()
        self.root.title("Drawing with Black Pen")

        # Create the canvas for drawing
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side=tk.LEFT)

        # Create a frame to hold the panel on the right side of the grid
        self.panel_frame = tk.Frame(self.root)
        self.panel_frame.pack(side=tk.LEFT)

        # Create the "Clear" button
        self.clear_button = tk.Button(self.panel_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.draw)

        # Create the "Export" button
        self.export_button = tk.Button(self.panel_frame, text="Export", command=self.calculate_averages)
        self.export_button.pack()

        # Start the Tkinter main loop
        self.root.mainloop()

    # Function to draw on the canvas using the black pen with a bigger brush
    def draw(self, event):
        x, y = event.x, event.y

        # Set a 5x5 square of pixels to black
        for dy in range(-9, 10):
            for dx in range(-9, 10):
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.canvas_width and 0 <= new_y < self.canvas_height:
                    self.grid[new_y][new_x] = 1
                    self.canvas.create_rectangle(new_x * self.pixel_size, new_y * self.pixel_size,
                                                 (new_x + 1) * self.pixel_size, (new_y + 1) * self.pixel_size,
                                                 fill="black")


    # Function to clear the canvas and reset the grid data
    def clear_canvas(self):
        self.canvas.delete("all")
        for y in range(self.canvas_height):
            for x in range(self.canvas_width):
                self.grid[y][x] = 0

    # Function to calculate the average values in each 10x10 sector
    def calculate_averages(self):
        averages = []
        for i in range(0, self.canvas_height, self.sector_size):
            for j in range(0, self.canvas_width, self.sector_size):
                sector_sum = 0
                for y in range(i, i + self.sector_size):
                    for x in range(j, j + self.sector_size):
                        sector_sum += self.grid[y][x]
                sector_avg = sector_sum / (self.sector_size ** 2)
                averages.append(sector_avg)

        addToDataset(averages, self.SetName, 1)

def addToDataset(array, name, legit = 0):
    data = loadData(name)
    if legit:
        data.append([array, [1, 0]])
    else:
        data.append([array, [0, 1]])
    storeData(data, name)

def storeData(data, name):
    buffer = io.BytesIO()

    # Pickle the data into the in-memory stream
    pickle.dump(data, buffer)

    # Compress the pickled data in-memory using gzip
    with gzip.open(f"{name}.pkl.gz", "wb") as f:
        f.write(buffer.getvalue())

def loadData(name):
    f = gzip.open(f'{name}.pkl.gz', 'rb')
    data = pickle.load(f, encoding="latin1")
    return data


if __name__ == "__main__":
    main("amogus")