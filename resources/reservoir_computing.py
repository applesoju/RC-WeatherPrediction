import matplotlib.pyplot as plt
import numpy as np


class SimpleESN:
    def __init__(self):
        self.data = None

        self.input_size = None
        self.output_size = None

        self.reservoir_size = None
        self.leaking_rate = None

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


def simple_esn(timeseries_filepath, training_length, test_length, initial_length=100):
    data = np.loadtxt(timeseries_filepath)

    plt.figure(1, figsize=(20, 12)).clear()
    plt.plot(data[:1000])

    plt.title("Sample of loaded data", fontsize=20)
    plt.xlabel("t", fontsize=16)
    plt.ylabel("P / \u0398", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    input_size = 1
    output_size = input_size
    reservoir_size = 1000
    leaking_rate = 0.25

    np.random.seed(64)

    input_weights = (np.random.rand(reservoir_size, 1 + input_size) - 0.5) * 1
    reservoir_weights = np.random.rand(reservoir_size, reservoir_size) - 0.5

    spectral_radius = max(abs(np.linalg.eig(reservoir_weights)[0]))
    print(f"Computed spectral radius: {spectral_radius}")

    reservoir_weights *= 1.25 / spectral_radius  # TODO: ?????

    x_out = np.zeros((1 + input_size + reservoir_size,
                      training_length - initial_length))
    yt = data[None, initial_length + 1: training_length + 1]
    x = np.zeros((reservoir_size, 1))

    for t in range(initial_length, training_length):
        u = data[t]
        x = (1 - leaking_rate) * x + \
            leaking_rate * np.tanh(np.dot(input_weights, np.vstack((1, u))) +
                                   np.dot(reservoir_weights, x))
        x_out[:, t - initial_length] = np.vstack((1, u, x))[:, 0]

    regr_coeff = 1e-8
    output_weights = np.linalg.solve(np.dot(x_out, x_out.T) +
                                     regr_coeff * np.eye(1 + input_size + reservoir_size),
                                     np.dot(x_out, yt.T)).T

    y_test = np.zeros((output_size, test_length))
    u = data[training_length]

    for t in range(test_length):
        x = (1 - leaking_rate) * x + \
            leaking_rate * np.tanh(np.dot(input_weights, np.vstack((1, u))) +
                                   np.dot(reservoir_weights, x))
        y = np.dot(output_weights, np.vstack((1, u, x)))
        y_test[:, t] = y
        u = y  # generative mode
        # u = data[training_length + 1]     # predictive mode

    error_length = 500
    mse = sum(np.square(data[training_length + 1: training_length + error_length + 1] -
                        y_test[0, 0: error_length])) / error_length
    print(f"MSE: {mse}")

    plt.figure(2, figsize=(20, 12)).clear()
    plt.plot(data[training_length + 1: training_length + test_length + 1], "g")
    plt.plot(y_test.T, "b")
    plt.title("Target and generated signals $y(n)$ starting at $n = 0$")
    plt.legend(["Target signal", "Predicted signal"])

    plt.figure(3, figsize=(20, 12)).clear()
    plt.plot(x_out[0: 20, 0: 200])
    plt.title(r"Some reservoir activations $\mathbf{x}(n)$")

    plt.figure(4, figsize=(20, 12)).clear()
    plt.bar(np.arange(1 + input_size + reservoir_size), output_weights[0].T)
    plt.title(r"Output weights $\mathbf{W}^{out}$")

    plt.show()
