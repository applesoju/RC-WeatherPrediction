import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class ModelEvaluation:
    def __init__(self, model, model_params):
        self.model = model
        self.model_params = model_params

        self.mse = None
        self.mae = None
        self.r2 = None

    def _reset_errors(self):
        self.mse = []
        self.mae = []
        self.r2 = []

    def _time_series_split(self, n_of_splits):
        data_len = len(self.model.data)
        split_len = data_len // (n_of_splits + 1)
        remaining_data_len = data_len % (n_of_splits + 1)

        first_split_len = split_len + remaining_data_len
        train_valid_splits = [(first_split_len + n * split_len, split_len) for n in range(n_of_splits)]

        return train_valid_splits

    def initialize_model(self):
        self.model.initialize_reservoir(*self.model_params)

    def generate_prediction(self, training_length, test_length):
        self.initialize_model()

        last_x = self.model.train(training_length=training_length)
        y_pred = self.model.predict(training_length=training_length,
                                    last_x=last_x,
                                    test_length=test_length)
        return y_pred

    def _save_metrics_to_file(self, filepath):
        try:
            metrics_df = pd.DataFrame({
                "mse": self.mse,
                "mae": self.mae,
                "r2": self.r2
            })
            metrics_df.to_csv(filepath, index=False)

        except OSError as err:
            print(f"Error: {err}\n"
                  f"The target directory does not exist.\n"
                  f"Skipping saving to file.\n")

    def cross_validate(self, n_of_splits, save_results_to_file=None, save_models_to_dir=None):
        print("Starting cross-validation...")

        self._reset_errors()
        ts_split = self._time_series_split(n_of_splits)

        i = 0
        for train_len, valid_len in ts_split:
            y_true = self.model.data[train_len: train_len + valid_len]
            y_pred = self.generate_prediction(training_length=train_len,
                                              test_length=valid_len)

            self.mse.append(mean_squared_error(y_true, y_pred))
            self.mae.append(mean_absolute_error(y_true, y_pred))
            self.r2.append(r2_score(y_true, y_pred))

            i += 1

            if save_models_to_dir is not None:
                print("Saving model for current split to file...")

                filename = f"model_{i:03d}.json"
                filepath = f"{save_models_to_dir}/{filename}"

                self.model.save_reservoir_to_file(filepath)

            print(f"Split {i} out of {len(ts_split)} done.\n")

        if save_results_to_file is not None:
            self._save_metrics_to_file(save_results_to_file)

    def plot_metrics(self):
        if self.mse is None:
            print(f"Error: metrics have not been generated.\n"
                  f"Skipping plotting metrics.")

        plt.figure(6, figsize=(20, 12)).clear()

        plt.subplot(311)
        plt.plot(self.mse)
        plt.title("Mean Square Error")

        plt.subplot(312)
        plt.plot(self.mae)
        plt.title("Mean Absolute Error")

        plt.subplot(313)
        plt.plot(self.r2)
        plt.title("R2 Score")

        plt.show()
