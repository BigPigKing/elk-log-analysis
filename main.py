import pandas as pd

from libs.data_preprocessing.raw_data_preprocessor import RawDataPreprocessor


GROUND_TRUTH_RAW_DATA = "raw_data/ground_truth.csv"


def main():
    raw_data_preprocessor: RawDataPreprocessor = RawDataPreprocessor(
        GROUND_TRUTH_RAW_DATA
    )

    processed_df: pd.DataFrame = raw_data_preprocessor.get_processed_dataframe()

    print(processed_df)


if __name__ == "__main__":
    main()
