from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .task2 import get_expert_predictions


def get_media_root():
    try:
        from django.conf import settings
        if settings.configured:
            return Path(settings.MEDIA_ROOT), settings.MEDIA_URL
    except Exception:
        pass
    fallback_dir = Path(__file__).resolve().parent.parent / "media"
    return fallback_dir, "/media/"


def generate_learning_curve_plot(budgets, random_accuracies, active_accuracies, ai_accuracy):
    media_root, media_url = get_media_root()
    plots_dir = media_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plots_dir / "task4_learning_curve.png"

    fig, ax = plt.subplots(figsize=(8, 4.8))

    # Show the difference from the AI baseline directly, so small accuracy
    # differences are not visually exaggerated by a tightly cropped accuracy axis.
    random_gains = [(value - ai_accuracy) * 100 for value in random_accuracies]
    active_gains = [(value - ai_accuracy) * 100 for value in active_accuracies]
    ax.plot(
        budgets,
        random_gains,
        "o-",
        color="#7f8c8d",
        label="Random Selection",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        budgets,
        active_gains,
        "s-",
        color="#2980b9",
        label="Active Selection (Uncertainty)",
        linewidth=2,
        markersize=6,
    )
    ax.axhline(
        0,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label=f"AI Baseline ({ai_accuracy * 100:.2f}%)",
    )

    for budget, value in zip(budgets, random_gains):
        ax.annotate(f"{value:.2f}", (budget, value), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9)
    for budget, value in zip(budgets, active_gains):
        ax.annotate(f"{value:.2f}", (budget, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)

    all_gains = random_gains + active_gains
    ax.set_ylim(min(all_gains) - 0.12, 0.12)
    ax.set_title("Team Accuracy Gain Relative to the AI Baseline", fontsize=13, fontweight="bold")
    plt.xlabel("Human Query Budget (B)", fontsize=11)
    plt.ylabel("Gain over AI (percentage points)", fontsize=11)
    plt.xticks(budgets)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    fig.tight_layout()

    fig.savefig(plot_file, dpi=130)
    plt.close(fig)

    return f"{media_url}plots/task4_learning_curve.png"


def run_task4(
    X_train,
    y_train,
    X_test,
    y_test_arr,
    model,
    ai_probs,
    y_pred,
    accuracy,
    expert_preds,
    expert_accuracy,
    budgets=(50, 100, 250, 500),
):
    # Use the fitted Task 1 vectorizer to convert the training and test articles to TF-IDF features
    tfidf = model.named_steps["tfidfvectorizer"]
    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Get the highest predicted class probability for each article to use as the AI confidence
    ai_train_probs = model.predict_proba(X_train)
    ai_train_conf = ai_train_probs.max(axis=1)
    ai_test_conf = ai_probs.max(axis=1)

    # Sort the training articles by AI confidence so the least confident articles are selected first
    uncertainty_ranked_indices = np.argsort(ai_train_conf)
    
    # Shuffle the training indices using a fixed seed so random selection can be repeated with the same order
    rng = np.random.RandomState(42)
    random_ranked_indices = rng.permutation(len(X_train))

    # Generate expert answers once so the same article has the same answer across budgets and strategies
    expert_train_preds, expert_train_in_topic = get_expert_predictions(
        X_train, y_train, seed=1
    )
    expert_correct_test = (expert_preds == y_test_arr).astype(int)

    results_by_budget = []
    random_accuracies = []
    active_accuracies = []

    for B in budgets:
        row = {"budget": B}

        for strategy in ("random", "active"):
            if strategy == "random":
                selected_indices = random_ranked_indices[:B]
            else:
                selected_indices = uncertainty_ranked_indices[:B]

            # Get the labels and stored expert answers for only the B selected training articles
            queried_labels = [y_train[i] for i in selected_indices]
            expert_queried_preds = expert_train_preds[selected_indices]
            
            # Compare the expert answers with the labels and assign 1 for a correct answer and 0 for an incorrect answer
            expert_correct_train = (expert_queried_preds == np.asarray(queried_labels)).astype(int)

            # Check whether the selected expert answers include both correct and incorrect predictions
            unique_classes = np.unique(expert_correct_train)
            if len(unique_classes) < 2:
                # If all answers are correct or all are incorrect, use their mean correctness as a constant prediction
                constant_rate = float(np.mean(expert_correct_train))
                pred_expert_competence = np.full(len(X_test), constant_rate)
            else:
                # Train the expert competence model using only the selected training articles and their answer correctness
                competence_model = LogisticRegression(max_iter=1000, random_state=42)
                competence_model.fit(X_train_tfidf[selected_indices], expert_correct_train)
                pred_expert_competence = competence_model.predict_proba(X_test_tfidf)[:, 1]

            # These values evaluate competence discovery only. Neither the hidden
            # specialty mask nor the test outcomes are used to select or route cases.
            specialty_query_count = int(np.sum(expert_train_in_topic[selected_indices]))
            specialty_query_share = specialty_query_count / B * 100
            competence_auc = roc_auc_score(
                expert_correct_test, pred_expert_competence
            )

            # Use the expert's answer when its predicted probability of being correct is higher than the AI confidence
            defer = pred_expert_competence > ai_test_conf
            team_preds = np.where(defer, expert_preds, y_pred)
            team_acc = round(float(accuracy_score(y_test_arr, team_preds)), 4)
            gain = round(team_acc - accuracy, 4)
            defer_rate = f"{float(np.mean(defer)) * 100:.2f}%"

            # Calculate the percentages of beneficial and harmful decisions among the deferred articles
            deferred_count = int(np.sum(defer))
            if deferred_count > 0:
                beneficial = defer & (y_pred != y_test_arr) & (expert_preds == y_test_arr)
                harmful = defer & (y_pred == y_test_arr) & (expert_preds != y_test_arr)
                beneficial_rate = f"{float(np.sum(beneficial)) / deferred_count * 100:.2f}%"
                harmful_rate = f"{float(np.sum(harmful)) / deferred_count * 100:.2f}%"
            else:
                beneficial_rate = "0.00%"
                harmful_rate = "0.00%"

            row[f"{strategy}_accuracy"] = team_acc
            row[f"{strategy}_accuracy_pct"] = f"{team_acc * 100:.2f}%"
            row[f"{strategy}_gain"] = gain
            row[f"{strategy}_gain_pp"] = f"{gain * 100:.2f}"
            row[f"{strategy}_deferral_rate"] = defer_rate
            row[f"{strategy}_beneficial_rate"] = beneficial_rate
            row[f"{strategy}_harmful_rate"] = harmful_rate
            row[f"{strategy}_specialty_query_count"] = specialty_query_count
            row[f"{strategy}_specialty_query_share"] = f"{specialty_query_share:.2f}%"
            row[f"{strategy}_competence_auc"] = f"{competence_auc:.3f}"

            if strategy == "random":
                random_accuracies.append(team_acc)
            else:
                active_accuracies.append(team_acc)

        results_by_budget.append(row)

    # Create a learning curve to compare random selection and active selection with the AI baseline
    learning_curve_plot_url = generate_learning_curve_plot(
        list(budgets), random_accuracies, active_accuracies, accuracy
    )

    return {
        "budgets": list(budgets),
        "task4_rows": results_by_budget,
        "learning_curve_plot_url": learning_curve_plot_url,
    }
