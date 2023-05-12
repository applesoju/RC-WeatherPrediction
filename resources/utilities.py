import os
import pandas as pd


def get_model_names(dirpath):
    if not os.path.exists(dirpath):
        raise FileNotFoundError("Provided directory does not exist.")

    names = os.listdir(dirpath)

    return names


def get_metric_from_gs(path, metric):
    models = get_model_names(path)

    metric_list = []
    for model in models:
        fullpath = f"{path}/{model}/errors.csv"
        errors_df = pd.read_csv(fullpath)
        metric_list.append(errors_df["r2"])

        return
