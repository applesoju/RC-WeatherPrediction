from os import path

import pandas as pd


class TmpTimeseries:
    def __init__(self):
        self.input_filename = None

        self.station_id = None
        self.station_name = None

        self.starting_date = None
        self.ending_date = None

        self.data = None

    def load_csv(self, filepath, column_names):
        csv_data = None

        self.input_filename = path.basename(filepath)

        if not path.exists(filepath):
            raise FileNotFoundError("File doesn't exist.")

        try:
            csv_data = pd.read_csv(filepath, dtype=str)
        except pd.errors.EmptyDataError:
            print("Error: Incorrect file format or can't read csv.")

        self.station_id = csv_data["STATION"].iloc[1]
        self.station_name = csv_data["NAME"].iloc[1]

        return csv_data[column_names]

    def process_tmp_dataframe(self, tmp_dataframe):
        dates = list(tmp_dataframe["DATE"])

        dates_with_time = [i.split("T") for i in dates]
        date, time = zip(*dates_with_time)

        year_month_day = [i.split("-") for i in date]
        year, month, day = zip(*year_month_day)

        hour_min_sec = [i.split(":") for i in time]
        hour, minute, second = zip(*hour_min_sec)

        temps = []
        for i in list(tmp_dataframe["TMP"]):
            temp_val, code = i.split(",")
            temps.append(temp_val if code == "1" else "+9999")

        year = [int(y) for y in year]
        month = [int(m) for m in month]
        day = [int(d) for d in day]
        hour = [int(h) for h in hour]
        minute = [int(m) for m in minute]
        second = [int(s) for s in second]
        temperature = [float(t) / 10 for t in temps]

        self.starting_date = (year[0], month[0], day[0])
        self.ending_date = (year[-1], month[-1], day[-1])

        processed_df = pd.DataFrame({
            "YEAR": year,
            "MONTH": month,
            "DAY": day,
            "HOUR": hour,
            "MINUTE": minute,
            "SECOND": second,
            "TMP": temperature
        })

        return processed_df

    def normalize_tmp_df(self):
        raise NotImplementedError