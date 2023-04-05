from os import path

import pandas as pd


def convert_columns(df_in, column_names, convert_to):
    new_df = pd.DataFrame(df_in)

    if type(column_names) is str:
        new_df[column_names] = df_in[column_names].str.replace(",", ".").astype(float)

    elif type(column_names) is list:
        for name in column_names:
            new_df[name] = df_in[name].str.replace(",", ".").astype(float)

    else:
        raise TypeError("Incorrect value for 'column_names' parameter.")

    return new_df


class WeatherData:
    def __init__(self):
        self.data = None
        self.columns = None

        self.input_filename = None
        self.station_id = None
        self.station_name = None

        self.starting_date = None
        self.ending_date = None

        self.cols_with_missing_data = None

    def load_csv(self, filepath):
        if not path.exists(filepath):
            raise FileNotFoundError("File doesn't exist.")

        try:
            self.data = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            print("Incorrect file format or can't read csv.")

        self.input_filename = path.basename(filepath)

        self.station_id = self.data["STATION"].iloc[1]
        self.station_name = self.data["NAME"].iloc[1]
        self.starting_date = self.data["DATE"].iloc[1]
        self.ending_date = self.data["DATE"].iloc[-1]

    def summary(self, print_columns=False, check_missing=False):
        print(f"Weather Data from: {self.input_filename}\n"
              f"Station ID: {self.station_id}\n"
              f"Station name: {self.station_name}\n"
              f"Number of attributes: {len(self.data.columns)}")

        if print_columns:
            print(f"Attribute names:")

            for i, name in enumerate(self.data.columns):
                print(f"{i:3d}. {name}")

        print(f"Starting date: {self.starting_date.split('T')[0]}")
        print(f"Ending date: {self.ending_date.split('T')[0]}")

        if check_missing:
            if not self.cols_with_missing_data:
                self.check_missing_values()

            print(f"Columns with missing data:")
            print(*self.cols_with_missing_data, sep=", ")

    def check_missing_values(self):
        missing_values = self.data.isnull().any()

        self.cols_with_missing_data = self.data.columns[missing_values].tolist()
