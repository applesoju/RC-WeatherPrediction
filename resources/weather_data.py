from os import path

import pandas as pd


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

    def column_summary(self, column_name):
        print(f"Summary of column '{column_name}':\n"
              f"{self.data[column_name].isnull().sum()} missing values.\n"
              f"{len(self.data[column_name][1].split(','))} elements in the column.\n"
              f"Sample data from the colum:")

        for sample in self.data[column_name].sample(5).tolist():
            print(sample)

    def get_timeseries(self, column_name, get_elem=None, convert_to=None, scale_by=None, save_to_file=None):
        column_data = self.data[column_name].tolist()

        try:
            timeseries = [i.split(",")[get_elem] for i in column_data] if get_elem is not None else column_data
        except AttributeError:
            print("Column type is not string and cannot be split.")
            timeseries = column_data

        try:
            timeseries = [convert_to(i) for i in timeseries if convert_to is not None]
        except ValueError:
            print(f"Column type cannot be converted to {convert_to.__name__}.")

        try:
            timeseries = [round(i * scale_by, 1) for i in timeseries if scale_by is not None]
        except TypeError:
            print(f"Column type does not allow multiplication.")

        if save_to_file is not None:

            if not path.exists(path.dirname(save_to_file)):
                raise OSError("Target directory does not exist.")

            with open(save_to_file, "w") as f:
                f.write("\n".join([str(i) for i in timeseries]))

        return timeseries
