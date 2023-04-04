from os import path

import pandas as pd


class WeatherData:
    def __init__(self):
        self.data = None

        self.input_filename = None
        self.station_name = None

        self.starting_date = None
        self.ending_date = None

    def load_csv(self, filepath):
        if not path.exists(filepath):
            raise FileNotFoundError("File doesn't exist.")

        try:
            self.data = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            print("Incorrect file format or can't read csv.")

        self.input_filename = path.basename(filepath)

        self.station_name = self.data["STATION"].iloc[1]
        self.starting_date = self.data["DATE"].iloc[1]
