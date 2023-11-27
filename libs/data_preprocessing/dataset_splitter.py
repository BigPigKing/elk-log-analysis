import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    def __init__(self, split_ratio: float = 0.7):
        self.split_ratio = split_ratio

    def set_split_ratio(self, split_ratio: float):
        self.split_ratio = split_ratio

    def split(self, featureX: pd.DataFrame, labelY: pd.DataFrame) -> List[pd.DataFrame]:
        trainX, testX, trainY, testY = train_test_split(
            featureX, labelY, train_size=self.split_ratio
        )

        return [trainX, testX, trainY, testY]


def main():
    pass


if __name__ == "__main__":
    main()
