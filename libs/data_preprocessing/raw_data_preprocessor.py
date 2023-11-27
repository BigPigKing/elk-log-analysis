import pandas as pd
from typing import List, Dict


DROP_COLUMN_NAMES: List[str] = [
    "id",
    "elkid",
    "sourceip",
    "targetip",
    "create_date",
    "update_date",
    "level_ai",
    "system_ai",
    "person_ai",
    "reason_ai",
    "reason_detail_ai",
    "isverify",
]

DROP_POST_COLUMN_NAMES: List[str] = [
    "id",
    "elkid",
    "sourceip",
    "targetip",
    "create_date",
    "update_date",
    "level",
    "system",
    "person",
    "reason",
    "reason_detail",
    "isverify",
]

POST_RENAMES: Dict[str, str] = {
    "level_ai": "level",
    "system_ai": "system",
    "person_ai": "person",
    "reason_ai": "reason",
    "reason_detail_ai": "reason_detail",
}


class RawDataPreprocessor:
    def __init__(
        self, raw_csv_path: str, drop_column_names: List[str] = DROP_COLUMN_NAMES
    ):
        self.raw_df: pd.DataFrame = pd.read_csv(raw_csv_path)
        self.drop_column_names = drop_column_names

        self.processed_df: pd.DataFrame = self._process_raw_df()

    def _process_raw_df(self) -> pd.DataFrame:
        # drop redundant columns
        processed_df = self.raw_df.drop(columns=self.drop_column_names)

        processed_df = processed_df[~pd.isna(processed_df["level"])]
        processed_df = processed_df[~pd.isna(processed_df["person"])]
        processed_df = processed_df[~pd.isna(processed_df["system"])]
        processed_df = processed_df[~pd.isna(processed_df["reason"])]
        processed_df = processed_df[~pd.isna(processed_df["reason_detail"])]

        return processed_df

    def get_processed_dataframe(self):
        return self.processed_df


class PostRawDataPreprocessor:
    def __init__(
        self, raw_csv_path: str, drop_column_names: List[str] = DROP_POST_COLUMN_NAMES
    ):
        self.raw_df: pd.DataFrame = pd.read_csv(raw_csv_path)
        self.drop_column_names = drop_column_names

        self.processed_df: pd.DataFrame = self._process_raw_df()

    def _process_raw_df(self) -> pd.DataFrame:
        # drop redundant columns
        processed_df = self.raw_df.drop(columns=self.drop_column_names)
        processed_df = processed_df.rename(columns=POST_RENAMES)

        processed_df = processed_df[~pd.isna(processed_df["level"])]
        processed_df = processed_df[~pd.isna(processed_df["person"])]
        processed_df = processed_df[~pd.isna(processed_df["system"])]
        processed_df = processed_df[~pd.isna(processed_df["reason"])]
        processed_df = processed_df[~pd.isna(processed_df["reason_detail"])]

        return processed_df

    def get_processed_dataframe(self):
        return self.processed_df


def main():
    pass


if __name__ == "__main__":
    main()
