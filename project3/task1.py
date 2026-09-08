import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline


def run_task1(X_train, y_train, X_test, y_test, model_file):
    # Load the saved model if it exists, otherwise train it on the complete training set and save it
    if model_file.exists():
        model = joblib.load(model_file)
    else:
        model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        model.fit(X_train, y_train)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_file)

    # Predict the class probabilities for the test articles and select the class with the highest probability
    ai_probs = model.predict_proba(X_test)
    y_pred = ai_probs.argmax(axis=1)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    report = classification_report(
        y_test, y_pred, target_names=["World", "Sports", "Business", "Sci/Tech"]
    )

    # Create a confusion matrix and convert it to an HTML table for display in the template
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_table = pd.DataFrame(
        conf_matrix, index=labels, columns=labels
    ).to_html()

    return {
        "model": model,
        "ai_probs": ai_probs,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "report": report,
        "conf_matrix_table": conf_matrix_table,
        "complexity_measure": len(model.named_steps["tfidfvectorizer"].vocabulary_),
    }


def main():
    # Load AG News and train the classifier using all labels in the training split
    dataset = load_dataset("fancyzhx/ag_news")

    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(dataset["train"]["text"], dataset["train"]["label"])

    # Calculate and print the accuracy on the test split
    accuracy = model.score(dataset["test"]["text"], dataset["test"]["label"])
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
