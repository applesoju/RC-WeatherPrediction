import matplotlib.pyplot as plt
import numpy as np


class SimpleESN:
    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.reservoir_size = None
        self.leaking_rate = None

    def loadtxt(self, filepath):
        raise NotImplementedError

    def plot_data(self, lenght, labels):
        raise NotImplementedError


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

    reservoir_weights *= 1.25 / spectral_radius     # TODO: ?????

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
        u = y                               # generative mode
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
