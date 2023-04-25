from os import path

import pandas as pd

DAYS_IN_MONTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
HOURS_RANGE = tuple(range(0, 24))


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

    def process_tmp_dataframe(self, tmp_df):
        dates = list(tmp_df["DATE"])

        dates_with_time = [i.split("T") for i in dates]
        date, time = zip(*dates_with_time)

        year_month_day = [i.split("-") for i in date]
        year, month, day = zip(*year_month_day)

        hour_min_sec = [i.split(":") for i in time]
        hour, minute, second = zip(*hour_min_sec)

        temps = []
        for i in list(tmp_df["TMP"]):
            temp_val, code = i.split(",")
            temps.append(temp_val if code == "1" else "+9999")

        year = [int(y) for y in year]
        month = [int(m) for m in month]
        day = [int(d) for d in day]
        hour = [int(h) for h in hour]
        minute = [int(m) for m in minute]
        temperature = [float(t) / 10 for t in temps]

        self.starting_date = (year[0], month[0], day[0])
        self.ending_date = (year[-1], month[-1], day[-1])

        processed_df = pd.DataFrame({
            "YEAR": year,
            "MONTH": month,
            "DAY": day,
            "HOUR": hour,
            "MINUTE": minute,
            "TMP": temperature
        })

        return processed_df

    def handle_missing_vals_in_tmp_df(self, tmp_df):    # TODO
        raise NotImplementedError

    def normalize_tmp_df(self, tmp_df):
        year_range = list(range(self.starting_date[0], self.ending_date[0] + 1))

        norm_df = pd.DataFrame(columns=["YEAR", "MONTH", "DAY", "HOUR", "TMP"])

        for year in year_range:
            dim = list(DAYS_IN_MONTHS)

            if year % 400 == 0 or year % 100 != 0 and year % 4 == 0:
                dim[1] += 1

            for month in range(len(DAYS_IN_MONTHS)):
                for day in range(dim[month]):
                    for hour in HOURS_RANGE:

                        tmp_val = tmp_df.loc[
                            (tmp_df["YEAR"] == year) &
                            (tmp_df["MONTH"] == month + 1) &
                            (tmp_df["DAY"] == day + 1) &
                            (tmp_df["HOUR"] == hour) &
                            (tmp_df["MINUTE"] == 0)
                        ]["TMP"]

                        if 999.9 in list(tmp_val):
                            tmp_val = [t for t in tmp_val if t != 999.9]
                        if len(tmp_val) == 0:
                            tmp_val = [999.9]
                        tmp_val = sum(tmp_val) / len(tmp_val)

                        new_row = pd.Series({
                            "YEAR": year,
                            "MONTH": month + 1,
                            "DAY": day + 1,
                            "HOUR": hour,
                            "TMP": tmp_val
                        })
                        norm_df = pd.concat([norm_df, new_row.to_frame().T])

                print(f"Year {year}, {month + 1} month done.")

        norm_df = norm_df.astype({
            "YEAR": int,
            "MONTH": int,
            "DAY": int,
            "HOUR": int,
            "TMP": float
        })

        return norm_df
