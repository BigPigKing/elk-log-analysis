import pandas as pd
import torch
from torch.utils.data import Dataset
from libs.training.model import TextEmbedder


class LevelDataset(Dataset):
    def __init__(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
        text_embedder: TextEmbedder,
    ):
        # Embedding is done in the forward pass of the model.
        # Store the DataFrame as is and handle embedding during training.
        self.feature_df = feature_df
        self.labels = torch.tensor(label_df.values, dtype=torch.long)
        self.text_embedder = text_embedder

    def __len__(self):
        return len(self.feature_df)

    def __getitem__(self, idx):
        # Extract text data for embedding and other features for this row
        message = self.feature_df.iloc[idx]["message"]
        message_detail = self.feature_df.iloc[idx]["message_detail"]
        hostname = self.feature_df.iloc[idx]["hostname"]

        # Embed the text data
        reason_features = self.text_embedder([message])
        reason_detail_features = self.text_embedder([message_detail])
        hostname_features = self.text_embedder([hostname])

        # Concatenate the features (adjust dim if necessary)
        feature_row = torch.cat(
            [reason_features, reason_detail_features, hostname_features], dim=1
        )

        # Get the corresponding label
        label_row = self.labels[idx]

        return feature_row.squeeze(), label_row.squeeze()


class TestDataset(Dataset):
    def __init__(
        self,
        feature_df: pd.DataFrame,
        text_embedder: TextEmbedder,
    ):
        # Embedding is done in the forward pass of the model.
        # Store the DataFrame as is and handle embedding during training.
        self.feature_df = feature_df
        self.text_embedder = text_embedder

    def __len__(self):
        return len(self.feature_df)

    def __getitem__(self, idx):
        # Extract text data for embedding and other features for this row
        message = self.feature_df.iloc[idx]["message"]
        message_detail = self.feature_df.iloc[idx]["message_detail"]
        hostname = self.feature_df.iloc[idx]["hostname"]

        # Embed the text data
        reason_features = self.text_embedder([message])
        reason_detail_features = self.text_embedder([message_detail])
        hostname_features = self.text_embedder([hostname])

        # Concatenate the features (adjust dim if necessary)
        feature_row = torch.cat(
            [reason_features, reason_detail_features, hostname_features], dim=1
        )

        return feature_row.squeeze()


def main():
    pass


if __name__ == "__main__":
    main()
