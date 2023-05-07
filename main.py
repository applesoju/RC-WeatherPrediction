from resources.model_evaluation import ModelEvaluation
from resources.reservoir_computing import SimpleESN
from resources.tmp_timeseries import TmpTimeseries

TRAINING_LEN = 5000
TEST_LENGTH = 2000
MODEL_PARAMS = [
    1000,  # Reservoir size
    0.25,  # Leaking rate
    1.25,  # Spectral radius
    64  # Seed for random initialization
]


def main():
    # tmp_ts = TmpTimeseries()
    # tmp_data = tmp_ts.load_csv(filepath="data/3289811.csv",
    #                            column_names=["DATE", "TMP"])

    # tmp_dataframe = tmp_ts.get_tmp_dataframe(tmp_df=tmp_data,
    #                                          restrain_year_range=(2000, 2022))
    # tmp_ts.save_csv("data/2000-2022/data.csv")
    # tmp_ts.save_txt("data/2000-2022/timeseries.txt")

    # tmp_timeseries = tmp_ts.get_tmp_timeseries(tmp_data)

    simple_esn = SimpleESN()

    # Load the data
    simple_esn.loadtxt(filepath="data/2000-2022/timeseries.txt")

    # # Plot loaded data
    simple_esn.plot_data(labels=["day", "temperature [C]"],
                         length=2000)

    # Initialize the reservoir
    simple_esn.initialize_reservoir(*MODEL_PARAMS)

    # Load model
    # simple_esn.load_reservoir_from_file("data/2000-2022/models/model_009.json")

    # # Perform training
    simple_esn.train(training_length=TRAINING_LEN)

    # Plot some activations of the reservoir
    simple_esn.plot_reservoir_activations(10, 10)

    # Plot trained output weigths
    simple_esn.plot_output_weights()

    # Predict data
    pred = simple_esn.predict(test_length=TEST_LENGTH,
                              training_length=TRAINING_LEN,
                              save_to_file="testing/prediction.txt")

    # Plot predicted data with target data and error
    simple_esn.plot_prediction_with_error(prediction=pred,
                                          training_length=TRAINING_LEN,
                                          test_length=TEST_LENGTH)

    # modeval = ModelEvaluation(model=simple_esn,
    #                           model_params=MODEL_PARAMS)

    # modeval.cross_validate(n_of_splits=10,
    #                        save_results_to_file="data/2000-2022/errors.csv",
    #                        save_models_to_dir="data/2000-2022/models",
    #                        save_models="all")
    #
    # modeval.plot_metrics()


    print("DONE")


if __name__ == "__main__":
    main()
