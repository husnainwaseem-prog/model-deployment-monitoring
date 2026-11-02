"""Train the iris classifier and save the deployment artifact."""

import os

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train(random_state=42):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25,
        random_state=random_state, stratify=iris.target,
    )
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "target_names": list(iris.target_names),
        # training baseline for drift detection
        "train_mean": {f: float(m) for f, m in
                       zip(feature_names, X_train.mean(axis=0))},
        "train_std": {f: float(s) for f, s in
                      zip(feature_names, X_train.std(axis=0))},
        "test_accuracy": accuracy,
        "n_train": len(X_train),
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "model.joblib")
    joblib.dump(artifact, path)
    print(f"test accuracy: {accuracy:.4f}  ->  saved {path}")
    return artifact


if __name__ == "__main__":
    train()
