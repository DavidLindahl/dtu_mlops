from pathlib import Path
from typing import Annotated, Optional

import typer
from torch.utils.data import Dataset
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MyDataset(Dataset):
    """My custom dataset for breast cancer data."""

    def __init__(self) -> None:
        """Initialize the dataset by loading breast cancer data."""
        # Load breast cancer dataset
        data = load_breast_cancer()
        x = data.data
        y = data.target

        # Split into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Scale the features
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        # Keep as numpy arrays for sklearn compatibility (y is already 1D from train_test_split)

    def __len__(self) -> int:
        """Return the length of the training dataset."""
        return len(self.x_train)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return torch.FloatTensor(self.x_train[index]), torch.FloatTensor([self.y_train[index]])

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        torch.save(torch.FloatTensor(self.x_train), output_folder / "x_train.pt")
        torch.save(torch.FloatTensor(self.x_test), output_folder / "x_test.pt")
        torch.save(torch.FloatTensor(self.y_train), output_folder / "y_train.pt")
        torch.save(torch.FloatTensor(self.y_test), output_folder / "y_test.pt")


def preprocess(
    output_folder: Annotated[
        Optional[str], typer.Option("--output-folder", "-o", help="Path to save processed data")
    ] = "data/processed",
) -> None:
    """Load and preprocess the breast cancer dataset."""
    print("Loading breast cancer dataset...")
    dataset = MyDataset()
    dataset.preprocess(Path(output_folder))
    print(f"Data saved to {output_folder}")


if __name__ == "__main__":
    typer.run(preprocess)
