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

        self.output_weigths = None
        self.x_out = None

    def loadtxt(self, filepath):
        print("Loading data from txt file...")
        try:
            self.data = np.loadtxt(filepath)
        except (ValueError, FileNotFoundError) as err:
            print(f"Error: {err}")

            match type(err).__name__:

                case "ValueError":
                    print(f"Filepath provided is not a string or the file is not a txt.")

                case "FileNotFoundError":
                    print(f"Provided file does not exist.")

                case _:
                    print(f"Unknown error occured.")

            print(f"No data loaded.\n")
            exit(-101)

    def plot_data(self, labels=None, length=None):
        if self.data is None:
            raise TypeError("Error: data has not been loaded.")

        if length is None:
            length = len(self.data)

        plt.figure(1, figsize=(20, 12)).clear()
        plt.plot(self.data[:length])
        plt.title("Sample of loaded data", fontsize=20)

        if labels is not None and len(labels) == 2:
            plt.xlabel(labels[0], fontsize=16)
            plt.ylabel(labels[1], fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

        plt.show()

    def initialize_reservoir(self, input_size, reservoir_size, leaking_rate, spectral_radius, seed=None):
        print("Initializing reservoir...")

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

    def train(self, training_length):
        x_out = np.zeros((1 + self.input_size + self.reservoir_size, training_length))
        y_target = self.data[None, 1: training_length + 1]
        x = np.zeros((self.reservoir_size, 1))

        for t in range(training_length):
            u = self.data[t]
            x = (1 - self.leaking_rate) * x + self.leaking_rate * \
                np.tanh(np.dot(self.input_weights, np.vstack((1, u))) +
                        np.dot(self.reservoir, x))

            x_out[:, t] = np.vstack((1, u, x))[:, 0]

        reg_coeff = 1e-8
        self.output_weigths = np.linalg.solve(
            np.dot(x_out, x_out.T) + reg_coeff * np.eye(1 + self.input_size + self.reservoir_size),
            np.dot(x_out, y_target.T)
        ).T

        self.x_out = x_out

        return x

    def predict(self, test_length, training_length, last_x, save_to_file=None):
        print("Model is generating a prediction...")

        y_test = np.zeros((self.output_size, test_length))
        u = self.data[training_length]
        x = last_x

        for t in range(test_length):
            x = (1 - self.leaking_rate) * x + self.leaking_rate * \
                np.tanh(np.dot(self.input_weights, np.vstack((1, u))) +
                        np.dot(self.reservoir, x))
            y = np.dot(self.output_weigths, np.vstack((1, u, x)))
            y_test[:, t] = y

            if t + 1 != test_length:
                u = self.data[training_length + t + 1]

        if save_to_file is not None:
            try:
                with open(save_to_file, "w") as f:
                    f.write("\n".join([str(i) for i in y_test[0]]))
            except OSError as err:
                print(f"Error: {err}\n"
                      f"The target directory does not exist.\n"
                      f"Skipping saving to file.\n")

        return y_test[0]

    def plot_reservoir_activations(self, xlen, ylen):
        plt.figure(2, figsize=(20, 12)).clear()
        plt.plot(self.x_out[0: xlen, 0: ylen])
        plt.title("Some reservoir activations")

        plt.show()

    def plot_output_weigths(self):
        plt.figure(3, figsize=(20, 12)).clear()
        plt.bar(np.arange(1 + self.input_size + self.reservoir_size), self.output_weigths[0].T)
        plt.title("Output weights")

        plt.show()

    def plot_prediction_with_error(self, prediction, training_length, test_length):

        err = np.square(self.data[training_length + 1:
                                  training_length + test_length + 1] - prediction[0])
        mse = sum(err) / test_length

        print(f"MSE: {mse}")

        plt.figure(4, figsize=(20, 12)).clear()
        plt.plot(self.data[training_length + 1:
                           training_length + test_length + 1], "g")
        plt.plot(prediction.T, "b")
        plt.title("Target and generated signals")
        plt.legend(["Target signal", "Predicted signal"])

        plt.figure(5, figsize=(20, 12)).clear()

        plt.subplot(211)
        plt.plot(err)
        plt.yscale("log")

        plt.subplot(212)
        plt.plot(err)
        plt.yscale("linear")

        plt.title(r"MSE")

        plt.show()
