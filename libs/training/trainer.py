import abc
import torch

from tqdm import tqdm
from sklearn.metrics import f1_score


class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self):
        return NotImplemented


class TextTrainer(Trainer):
    def __init__(self, train_model: torch.nn.Module, device):
        super(TextTrainer, self).__init__()
        self.train_model = train_model
        self.device = device

    def _compute_metrics(self, total_labels, total_predicts):
        # Convert to NumPy arrays
        total_labels_np = total_labels.cpu().numpy()
        total_predicts_np = total_predicts.cpu().numpy()

        # Calculate F1 score
        f1 = f1_score(
            total_labels_np, total_predicts_np, average="weighted"
        )  # 'weighted' can be changed based on your need

        return f1

    def _fit_valid(self, valid_data_loader: torch.utils.data.DataLoader):
        num_of_batch = 0
        total_loss = 0.0
        total_labels = []
        total_predicts = []

        for batch_idx, (batchX, batchY) in enumerate(valid_data_loader):
            batchX = batchX.to(self.device)
            batchY = batchY.to(self.device)
            output_dict = self.train_model.forward(batchX, batchY)

            num_of_batch += 1
            total_labels.append(batchY)
            total_predicts.append(output_dict["predicts"])
            total_loss += output_dict["cross_entropy_loss"].item()

        total_labels = torch.cat(total_labels, 0)
        total_predicts = torch.cat(total_predicts, 0)

        avg_loss = total_loss / num_of_batch
        avg_acc = torch.true_divide(
            torch.sum(total_labels == total_predicts), total_labels.shape[0]
        )
        f1 = self._compute_metrics(total_labels, total_predicts)

        return avg_loss, avg_acc, f1

    def _fit_raw_valid(self, no_label_data_loader: torch.utils.data.DataLoader):
        num_of_batch = 0
        total_predicts = []

        for batch_idx, batchX in enumerate(no_label_data_loader):
            batchX = batchX.to(self.device)
            batch_predicts = self.train_model.forward(batchX)

            num_of_batch += 1
            total_predicts.append(batch_predicts)

        total_predicts = torch.cat(total_predicts, 0)

        return total_predicts

    def _fit_train(self, train_data_loader: torch.utils.data.DataLoader):
        num_of_batch = 0
        total_loss = 0.0
        total_labels = []
        total_predicts = []

        for batch_idx, (batchX, batchY) in enumerate(tqdm(train_data_loader)):
            batchX = batchX.to(self.device)
            batchY = batchY.to(self.device)
            output_dict = self.train_model.forward(batchX, batchY)
            self.train_model.optimize(output_dict["cross_entropy_loss"])

            num_of_batch += 1
            total_labels.append(batchY)
            total_predicts.append(output_dict["predicts"])
            total_loss += output_dict["cross_entropy_loss"].item()

        self.train_model.optimizer.zero_grad()

        total_labels = torch.cat(total_labels, 0)
        total_predicts = torch.cat(total_predicts, 0)

        avg_loss = total_loss / num_of_batch
        avg_acc = torch.true_divide(
            torch.sum(total_labels == total_predicts), total_labels.shape[0]
        )

        loss_dict = {
            "avg_loss": avg_loss,
        }
        f1 = self._compute_metrics(total_labels, total_predicts)

        return loss_dict, avg_acc, f1

    def predict(self, test_data_loader: torch.utils.data.DataLoader):
        self.train_model.eval()
        with torch.no_grad():
            test_avg_loss, test_avg_acc = self._fit_valid(test_data_loader)

        print("Testing Loss             : {:.5f}".format(test_avg_loss))
        print("Testing Acc              : {:.5f}".format(test_avg_acc))
        print("----------------------------------------------")

    def raw_predict(self, no_label_data_loader: torch.utils.data.DataLoader):
        self.train_model.eval()
        with torch.no_grad():
            total_predicts = self._fit_raw_valid(no_label_data_loader)

        return total_predicts

    def fit(
        self,
        epochs: int,
        train_data_loader: torch.utils.data.DataLoader,
        valid_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader = None,
    ):
        for epoch in tqdm(range(epochs)):
            # Do training
            self.train_model.train()
            loss_dict, train_avg_acc, train_f1 = self._fit_train(train_data_loader)

            # Do validation
            self.train_model.eval()
            with torch.no_grad():
                valid_avg_loss, valid_avg_acc, valid_f1 = self._fit_valid(
                    valid_data_loader
                )

            # Do testing
            self.train_model.eval()
            with torch.no_grad():
                test_avg_loss, test_avg_acc, test_f1 = self._fit_valid(test_data_loader)

        # Print metrics
        print(f"Epochs: {epoch}")
        print(
            f"Training Loss: {loss_dict['avg_loss']:.5f}, Acc: {train_avg_acc:.5f}, F1: {train_f1:.5f}"
        )
        print(
            f"Validation Loss: {valid_avg_loss:.5f}, Acc: {valid_avg_acc:.5f}, F1: {valid_f1:.5f}"
        )
        if test_data_loader is not None:
            print(
                f"Testing Loss: {test_avg_loss:.5f}, Acc: {test_avg_acc:.5f}, F1: {test_f1:.5f}"
            )

    def save(self, file_path: str):
        torch.save(self.train_model.state_dict(), file_path + ".pkl")

    def load(self, file_path: str):
        self.train_model.load_state_dict(torch.load(file_path + ".pkl"))


def main():
    pass


if __name__ == "__main__":
    pass
