import pickle

import typer

from sklearn.metrics import accuracy_score, classification_report

from data import MyDataset

app = typer.Typer()


@app.command()
def evaluate(model_path: str):
    """Evaluate the model."""
    dataset = MyDataset()
    x_test, y_test = dataset.x_test, dataset.y_test
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report


def main():
    """Main entry point for the evaluate script."""
    app()


if __name__ == "__main__":
    main()
