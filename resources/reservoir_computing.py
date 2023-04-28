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

    def loadtxt(self, filepath, missing_values=None, save_to_file=None):
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

        if save_to_file is not None:
            try:
                with open(save_to_file, "w") as f:
                    f.write("\n".join([str(i) for i in self.data]))
            except OSError as err:
                print(f"Error: {err}\n"
                      f"The target directory does not exist.\n"
                      f"Skipping saving to file.\n")

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

    def train(self, initial_length, training_length):
        x_out = np.zeros((1 + self.input_size + self.reservoir_size, training_length - initial_length))
        y_target = self.data[None, initial_length + 1: training_length + 1]
        x = np.zeros((self.reservoir_size, 1))

        for t in range(initial_length, training_length):
            u = self.data[t]
            x = (1 - self.leaking_rate) * x + self.leaking_rate * \
                np.tanh(np.dot(self.input_weights, np.vstack((1, u))) +
                        np.dot(self.reservoir, x))

            x_out[:, t - initial_length] = np.vstack((1, u, x))[:, 0]

        reg_coeff = 1e-8
        self.output_weigths = np.linalg.solve(
            np.dot(x_out, x_out.T) + reg_coeff * np.eye(1 + self.input_size + self.reservoir_size),
            np.dot(x_out, y_target.T)
        ).T

        self.x_out = x_out

        return x

    def predict(self, test_length, training_length, initial_length, last_x, mode, save_to_file=None):
        y_test = np.zeros((self.output_size, test_length))
        u = self.data[training_length + initial_length]
        x = last_x
        u1 = u2 = u

        for t in range(test_length):
            x = (1 - self.leaking_rate) * x + self.leaking_rate * \
                np.tanh(np.dot(self.input_weights, np.vstack((1, u))) +
                        np.dot(self.reservoir, x))
            y = np.dot(self.output_weigths, np.vstack((1, u, x)))
            y_test[:, t] = y

            match mode:
                case "g":
                    u = y
                case "p":
                    u = self.data[training_length + initial_length + t + 1]
                case _:
                    raise ValueError("Error: parameter 'mode' has to be either 'g' for 'generative'"
                                     "or 'p' for 'predictive.")

        if save_to_file is not None:
            try:
                with open(save_to_file, "w") as f:
                    f.write("\n".join([str(i) for i in y_test[0]]))
            except OSError as err:
                print(f"Error: {err}\n"
                      f"The target directory does not exist.\n"
                      f"Skipping saving to file.\n")

        return y_test

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

    def plot_prediction_with_error(self, prediction, initial_length, training_length, test_length):

        err = np.square(self.data[training_length + initial_length + 1:
                                  training_length + initial_length + test_length + 1] - prediction[0])
        mse = sum(err) / test_length

        print(f"MSE: {mse}")

        plt.figure(4, figsize=(20, 12)).clear()
        plt.plot(self.data[training_length + initial_length + 1:
                           training_length + initial_length + test_length + 1], "g")
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

