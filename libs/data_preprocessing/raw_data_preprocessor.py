import pandas as pd


class RawDataPreprocessor:
    def __init__(self, raw_csv_path: str):
        self.raw_df: pd.DataFrame = pd.read_csv(raw_csv_path)
        self.processed_df: pd.DataFrame = self._process_raw_df()

    def _process_raw_df(self) -> pd.DataFrame:
        print(self.raw_df)

        return self.raw_df

    def get_processed_dataframe(self):
        return self.processed_df


def main():
    pass


if __name__ == "__main__":
    main()
