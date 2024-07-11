import tkinter as tk
from . import Network
import random

class DrawingApp:
    def __init__(self, NNname):
        # Initialize the canvas dimensions and pixel size
        self.canvas_width = 280
        self.canvas_height = 280
        self.pixel_size = 1
        self.sector_size = 10  # Size of each sector
        self.NNname = NNname
        self.NeuralNetwork = Network.Network(name=NNname)
        self.NNoutputs = Network.Network(name=self.NNname).sizes[-1]

        # Initialize the 280x280 grid as a list of lists (all set to 0 initially)
        self.grid = [[0 for _ in range(self.canvas_width)] for _ in range(self.canvas_height)]

        # Initialize the values for the panel on the left side of the grid
        self.panel_values = [0] * self.NNoutputs

        # Create the GUI
        self.root = tk.Tk()
        self.root.title("Drawing with Black Pen")

        # Create the canvas for drawing
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side=tk.LEFT)

        # Create a frame to hold the panel on the right side of the grid
        self.panel_frame = tk.Frame(self.root)
        self.panel_frame.pack(side=tk.LEFT)

        # Create labels for the values on the panel
        self.panel_labels = []
        for i in range(self.NNoutputs):
            label = tk.Label(self.panel_frame, text=f"{i}: 0")
            label.pack()
            self.panel_labels.append(label)

        # Create the "Clear" button
        self.clear_button = tk.Button(self.panel_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.draw)

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

        if random.randint(0, 7) == 0:
                self.calculate_averages()

    # Function to clear the canvas and reset the grid data
    def clear_canvas(self):
        self.canvas.delete("all")
        for y in range(self.canvas_height):
            for x in range(self.canvas_width):
                self.grid[y][x] = 0
        self.set_panel_values([0] * self.NNoutputs)

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

        self.set_panel_values(self.evaluate(averages, self.NNname))

    # Function to set the panel values based on the provided values
    def set_panel_values(self, values):
        if len(values) == self.NNoutputs:
            self.panel_values = values
            for i in range(self.NNoutputs):
                label = self.panel_labels[i]
                label.config(text=f"{i}: {round(values[i], 2)}%")

    def evaluate(self, input, NNname):
        result = self.NeuralNetwork.feedforward(input).flatten()
        sumN = sum(result)
        percentages = [100 * (x / sumN) for x in result]
        return percentages

# Create an instance of the DrawingApp class
if __name__ == "__main__":
    DrawingApp("amogus")