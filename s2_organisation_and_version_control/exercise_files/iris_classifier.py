import typer
from typing import Optional, Annotated
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = typer.Typer()


# Load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

@app.command()
def train(output: Annotated[Optional[str], typer.Option("--output", "-o")] = None):
    """Train the model."""

    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel="linear", random_state=42)
    model.fit(x_train, y_train)

    if output is not None:
        with open(output, "wb") as f:
            pickle.dump(model, f)
    
@app.command()
def evaluate(model_path: str):
    """Evaluate the model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report


# this "if"-block is added to enable the script to be run from the command line
if __name__ == "__main__":
    app()
    