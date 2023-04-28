from resources.tmp_timeseries import TmpTimeseries
from resources.reservoir_computing import SimpleESN

TRAINING_LEN = 5000
TEST_LENGTH = 1000


def main():
    # tmp_ts = TmpTimeseries()
    # tmp_data = tmp_ts.load_csv(filepath="testing/07003099999.csv",
    #                            column_names=["DATE", "TMP"])
    #
    # tmp_dataframe = tmp_ts.get_tmp_dataframe(tmp_data)
    # tmp_ts.save_csv("testing/out_test.csv")
    # tmp_ts.save_txt("testing/timeseries.txt")
    #
    # tmp_timeseries = tmp_ts.get_tmp_timeseries(tmp_data)

    simple_esn = SimpleESN()

    # Load the data
    simple_esn.loadtxt(filepath="testing/timeseries.txt")

    # # Plot loaded data
    simple_esn.plot_data(labels=["day", "temperature [\u2103]"],
                         length=2000)

    # Initialize the reservoir
    simple_esn.initialize_reservoir(input_size=1,
                                    reservoir_size=1000,
                                    leaking_rate=0.25,
                                    spectral_radius=1.25,
                                    seed=64)

    # Perform training
    res = simple_esn.train(training_length=TRAINING_LEN)

    # Plot some activations of the reservoir
    # simple_esn.plot_reservoir_activations(10, 10)

    # Plot trained output weigths
    # simple_esn.plot_output_weigths()

    # Predict data
    pred = simple_esn.predict(test_length=TEST_LENGTH,
                              training_length=TRAINING_LEN,
                              last_x=res,
                              mode="p",
                              save_to_file="testing/prediction.txt")

    # Plot predicted data with target data and error
    simple_esn.plot_prediction_with_error(prediction=pred,
                                          training_length=TRAINING_LEN,
                                          test_length=TEST_LENGTH)

    print("DONE")


if __name__ == "__main__":
    main()
