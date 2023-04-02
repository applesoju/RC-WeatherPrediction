from resources.reservoir_computing import simple_esn
from resources.weather_data_processing import weather_csv_to_timeseries, convert_columns_to_float, rescale_columns


def main():
    station, df = weather_csv_to_timeseries(filepath="testing/07003099999.csv",
                                            attribute_name="TMP")
    df = convert_columns_to_float(df_in=df,
                                  column_names="TMP")
    df = rescale_columns(df_in=df,
                         column_names="TMP",
                         scale_by=0.1)

    print("DONE")


if __name__ == "__main__":
    # main()
    simple_esn(timeseries_filepath="MackeyGlass_t17.txt",
               training_length=0,
               test_length=0)
