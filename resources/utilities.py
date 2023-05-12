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
        metric_list.append(errors_df[metric].tolist())

    metric_df = pd.DataFrame(metric_list, index=models, columns=[str(i + 1) for i in range(len(metric_list[0]))])

    return metric_df


def sort_df_by_split(df, split, desc=True, get_best=None):
    sorted_df = df.sort_values(by=[*split], ascending=not desc)

    if get_best is not None:
        sorted_df = sorted_df.head(get_best)

    return sorted_df


def sort_df_by_mean(df, desc=True, get_best=None):
    row_means = df.mean(axis=1)

    if desc:
        reverse_idx = row_means.argsort()[::-1]
        sorted_df = df.iloc[reverse_idx]

    else:
        sorted_df = df.iloc[row_means.argsort()]

    if get_best is not None:
        sorted_df = sorted_df.head(get_best)

    return sorted_df
