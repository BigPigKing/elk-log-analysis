import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sentence_transformers import SentenceTransformer


class TextEmbedder(nn.Module):
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()

        self.embedder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

    def forward(self, text):
        with torch.no_grad():
            text_vector = self.embedder.encode(text, convert_to_tensor=True)

        return text_vector


class LevelModel(nn.Module):
    def __init__(self, text_embedder: nn.Module, output_size: int):
        super(LevelModel, self).__init__()
        self.embedder = text_embedder
        self.linear1 = nn.Linear(1152, 256)
        self.linear2 = nn.Linear(256, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.parameters(), lr=0.001)

    def forward(self, batchX, batchY=None):
        # Embedding and concatenation of text features should be handled before this step
        # Assuming batchX is already in the correct shape [batch_size, 1152]

        # Forward pass through the network
        x = F.relu(self.linear1(batchX))
        logits = self.linear2(x)

        # If batchY is provided, calculate loss
        if batchY is not None:
            loss = self.criterion(logits, batchY)
            return {
                "predicts": torch.argmax(logits, dim=1),  # Class predictions
                "cross_entropy_loss": loss,
            }

        # For prediction, only return the logits or class predictions
        return torch.argmax(logits, dim=1)

    def optimize(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


def main():
    pass


if __name__ == "__main__":
    main()
