import resources.weather_data_processing as wdp


def main():
    station, df = wdp.weather_csv_to_timeseries(filepath="testing/07003099999.csv",
                                                attribute_name="TMP")
    df = wdp.convert_columns_to_float(df_in=df,
                                      column_names="TMP")
    df = wdp.rescale_columns(df_in=df,
                             column_names="TMP",
                             scale_by=0.1)

    print("DONE")


if __name__ == "__main__":
    main()
