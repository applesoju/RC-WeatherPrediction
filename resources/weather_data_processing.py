from os.path import exists

import pandas as pd


def weather_csv_to_timeseries(filepath, attribute_name):
    if not exists(filepath):
        raise FileNotFoundError("File doesn't exist.")

    try:
        loaded_csv = pd.read_csv(filepath, dtype=str)
    except EmptyDataError:
        print("Provided file is not a csv file.")
        exit(1)

    timeseries_df = loaded_csv[["DATE", attribute_name]]

    return *loaded_csv["STATION"].unique(), timeseries_df


def convert_columns_to_float(df_in, column_names):
    new_df = pd.DataFrame(df_in)

    if type(column_names) is str:
        new_df[column_names] = df_in[column_names].str.replace(",", ".").astype(float)

    elif type(column_names) is list:
        for name in column_names:
            new_df[name] = df_in[name].str.replace(",", ".").astype(float)

    else:
        raise TypeError("Incorrect value for 'column_names' parameter.")

    return new_df


def rescale_columns(df_in, column_names, scale_by):
    new_df = pd.DataFrame(df_in)

    if type(column_names) is str:
        new_df[column_names] = df_in[column_names].apply(lambda x: x * scale_by)

    elif type(column_names) is list:
        for name in column_names:
            new_df[name] = df_in[column_names].apply(lambda x: x * scale_by)

    else:
        raise TypeError("Incorrect value for 'column_names' parameter.")

    return new_df
