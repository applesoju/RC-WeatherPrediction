import itertools
import os.path
import subprocess
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set()


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

        self.model.train(training_length=training_length)
        y_pred = self.model.predict(test_length=test_length)

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
                  f"Skipping saving to file.")

    def cross_validate(self, n_of_splits, save_results_to_file=None, save_models_to_dir=None, save_models="last"):
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

            if save_models_to_dir is not None and save_models == "all":
                print("Saving model for current split to file...")

                filename = f"model_{i:03d}.json"
                filepath = f"{save_models_to_dir}/{filename}"

                self.model.save_reservoir_to_file(filepath)

            print(f"Split {i} out of {len(ts_split)} done.\n")

        if save_models_to_dir is not None and save_models == "last":
            print("Saving last model to file...")
            self.model.save_reservoir_to_file(f"{save_models_to_dir}/model.json")

        if save_results_to_file is not None:
            self._save_metrics_to_file(save_results_to_file)

    def plot_metrics(self):
        if self.mse is None:
            print(f"Error: metrics have not been generated.\n"
                  f"Skipping plotting metrics.")

        plt.figure(6, figsize=(20, 12)).clear()
        sns.lineplot(self.mse)
        plt.title("Mean Square Error")
        for i, val in enumerate(self.mse):
            label = round(val, 2)
            plt.annotate(label, (i, val))

        plt.figure(7, figsize=(20, 12)).clear()
        sns.lineplot(self.mae)
        plt.title("Mean Absolute Error")
        for i, val in enumerate(self.mae):
            label = round(val, 2)
            plt.annotate(label, (i, val))

        plt.figure(8, figsize=(20, 12)).clear()
        sns.lineplot(self.r2)
        plt.title("R2 Score")

        for i, val in enumerate(self.r2):
            label = round(val, 2)
            plt.annotate(label, (i, val))

        plt.show()

    def grid_search(self, params, dirpath):
        combos = list(itertools.product(*list(params.values())))

        if not os.path.exists(dirpath):
            raise FileNotFoundError("Directory not found.\n"
                                    "Aborting grid search.")

        for combo in combos:
            print(f"Testing model with parameters:\n"
                  f"    Reservoir size: {combo[0]}\n"
                  f"    Leaking rate: {combo[1]}\n"
                  f"    Rho: {combo[2]}\n")

            dirname = "-".join([str(c) for c in combo]).replace(".", "_")
            full_path = f"{dirpath}/{dirname}"

            if not os.path.exists(full_path):
                subprocess.run(["mkdir", full_path.replace("/", "\\")], shell=True)

            self.model_params = list(combo) + [64]

            self.cross_validate(n_of_splits=5,
                                save_results_to_file=f"{full_path}/errors.csv",
                                save_models_to_dir=full_path)

    def variable_params_test(self, params, dirpath):
        rs_const, lr_const, rho_const = self.model_params

        if not os.path.exists(dirpath):
            raise FileNotFoundError("Directory not found.\n"
                                    "Aborting test.")

        rs_var = [params["reservoir_sizes"], [lr_const], [rho_const]]
        rs_combos = list(itertools.product(*rs_var))

        lr_var = [[rs_const], params["leaking_rates"], [rho_const]]
        lr_combos = list(itertools.product(*lr_var))

        rho_var = [[rs_const], [lr_const], params["spectral_radiuses"]]
        rho_combos = list(itertools.product(*rho_var))

        for rsc in rs_combos:
            print(f"Testing model with constants:\n"
                  f"    Leaking rate: {rsc[1]}\n"
                  f"    Rho: {rsc[2]}\n"
                  f"And variable reservoir size: {rsc[0]}")

            dirname = f"var_reservoir-size/{str(rsc[0])}"
            full_path = f"{dirpath}/{dirname}"
            if not os.path.exists(full_path):
                subprocess.run(["mkdir", full_path.replace("/", "\\")], shell=True)

            self.model_params = list(rsc) + [64]

            start_time = time.time()

            self.cross_validate(n_of_splits=5,
                                save_results_to_file=f"{full_path}/metrics.csv",
                                save_models_to_dir=full_path)

            end_time = time.time()
            elapsed = end_time - start_time

            with open(f"{full_path}/time.txt", "w") as f:
                f.write(str(elapsed))

        for lrc in lr_combos:
            print(f"Testing model with constants:\n"
                  f"    Reservoir size: {lrc[0]}\n"
                  f"    Rho: {lrc[2]}\n"
                  f"And variable leaking rate: {lrc[1]}")

            dirname = f"var_leaking-rate/{str(lrc[1]).replace('.', '_')}"
            full_path = f"{dirpath}/{dirname}"
            if not os.path.exists(full_path):
                subprocess.run(["mkdir", full_path.replace("/", "\\")], shell=True)

            self.model_params = list(lrc) + [64]

            self.cross_validate(n_of_splits=5,
                                save_results_to_file=f"{full_path}/metrics.csv",
                                save_models_to_dir=full_path)

        for rhoc in rho_combos:
            print(f"Testing model with constants:\n"
                  f"    Reservoir size: {rhoc[0]}\n"
                  f"    Leaking rate: {rhoc[1]}\n"
                  f"And variable spectral radius: {rhoc[2]}")

            dirname = f"var_spectral-radius/{str(rhoc[2]).replace('.', '_')}"
            full_path = f"{dirpath}/{dirname}"
            if not os.path.exists(full_path):
                subprocess.run(["mkdir", full_path.replace("/", "\\")], shell=True)

            self.model_params = list(rhoc) + [64]

            self.cross_validate(n_of_splits=5,
                                save_results_to_file=f"{full_path}/metrics.csv",
                                save_models_to_dir=full_path)

        return
