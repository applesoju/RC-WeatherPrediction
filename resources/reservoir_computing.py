import matplotlib.pyplot as plt
import numpy as np


class SimpleESN:
    def __init__(self):
        self.data = None

        self.input_size = None
        self.output_size = None

        self.reservoir_size = None
        self.leaking_rate = None

        self.input_weights = None
        self.reservoir = None
        self.spectral_radius = None

    def loadtxt(self, filepath, missing_values=None):
        try:
            self.data = np.loadtxt(filepath)
        except (ValueError, FileNotFoundError) as err:
            print(f"Error: {err}\n"
                  f"File does not exist or is not a txt.\n"
                  f"No data loaded.\n")
            exit(101)

        if missing_values is not None:
            if type(missing_values) is not int and type(missing_values) is not float:
                raise TypeError("Error: parameter 'missing_values' has to be numeric.")

            for i, val in enumerate(self.data):
                if val == missing_values:

                    next_ind = i + 1
                    next_value = self.data[next_ind]

                    while next_value == missing_values:
                        next_ind += 1
                        next_value = self.data[next_ind]

                    new_values = np.linspace(self.data[i - 1], self.data[next_ind], next_ind - i + 2)
                    self.data[i: next_ind] = new_values[1: -1]

    def plot_data(self, labels=None, length=None):
        if self.data is None:
            raise TypeError("Error: data has not been loaded.")

        if length is None:
            length = len(self.data)

        plt.figure(figsize=(20, 12)).clear()
        plt.plot(self.data[:length])
        plt.title("Sample of loaded data", fontsize=20)

        if labels is not None and len(labels) == 2:
            plt.xlabel(labels[0], fontsize=16)
            plt.ylabel(labels[1], fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

        plt.show()

    def initialize_reservoir(self, input_size, reservoir_size, leaking_rate, spectral_radius, seed=None):
        self.input_size = input_size
        self.output_size = self.input_size
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius

        if seed is not None:
            np.random.seed(seed)

        self.input_weights = np.random.rand(self.reservoir_size, 1 + input_size) - 0.5
        self.reservoir = np.random.rand(reservoir_size, reservoir_size) - 0.5

        rho = max(abs(np.linalg.eig(self.reservoir)[0]))
        self.reservoir *= self.spectral_radius / rho

        print("ok")

    def train(self):
        raise NotImplementedError
