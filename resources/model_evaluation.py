import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class ModelEvaluation:
    def __init__(self, model, target_ts=None):
        self.model = model

        match type(target_ts).__name__:

            case "NoneType":
                print("Warning: no timeseries provided.")

            case "str":
                self.loadtxt(target_ts)

            case "list":
                self.target_ts = target_ts

            case _:
                self.target_ts = None
                print("Error: provided parameter is not a list or a filepath.")

    def loadtxt(self, filepath):
        try:
            self.target_ts = np.loadtxt(filepath)
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

    def generate_prediction(self, training_lenght, test_lenght):
        self.model.train(training_length=training_lenght)
        y_pred = self.model.predict(training_length=training_lenght,
                                    test_lenght=test_lenght)
        return y_pred

    def cross_validate(self, n_of_splits):
        ts_split = TimeSeriesSplit(n_splits=n_of_splits)

        for train_idx, valid_idx in ts_split.split(self.target_ts):
            train, valid = self.target_ts.iloc[train_idx], self.target_ts.iloc[valid_idx]
            raise NotImplementedError
