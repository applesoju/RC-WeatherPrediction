import matplotlib.pyplot as plt
import numpy as np


def simple_esn(timeseries_filepath, training_length, test_length, initial_length=100):
    data = np.loadtxt(timeseries_filepath)

    plt.figure(figsize=(20, 12)).clear()
    plt.plot(data[:1000])

    plt.title("Sample of loaded data")
    plt.xlabel("t", fontsize=16)
    plt.ylabel("P / \u03C3")

    plt.show()
