from model import Model
from data import MyDataset

import typer
from typing import Optional, Annotated
import pickle

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")

dataset = MyDataset()
x_train , y_train = dataset.x_train, dataset.y_train

@train_app.command()
def svm(kernel: str = "linear", output: Annotated[Optional[str], typer.Option("--output", "-o")] = "models/svm_model.pkl"):
    """Train an SVM model."""

    print(f"Starting training of SVM model with kernel='{kernel}'...")
    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel=kernel, random_state=42)
    print("Fitting SVM model to training data...")
    model.fit(x_train, y_train)
    print("SVM model training complete.")

    if output is not None:
        print(f"Saving SVM model to '{output}'...")
        with open(output, "wb") as f:
            pickle.dump(model, f)
        print("Model saved.")

@train_app.command()
def knn(n_neighbors: int = 5, output: Annotated[Optional[str], typer.Option("--output", "-o")] = "models/knn_model.pkl"):
    """Train a KNN model."""

    print(f"Starting training of KNN model with n_neighbors={n_neighbors}...")
    # Train a KNN model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    print("Fitting KNN model to training data...")
    model.fit(x_train, y_train)
    print("KNN model training complete.")

    if output is not None:
        print(f"Saving KNN model to '{output}'...")
        with open(output, "wb") as f:
            pickle.dump(model, f)
        print("Model saved.")

# this "if"-block is added to enable the script to be run from the command line
if __name__ == "__main__":
    app()