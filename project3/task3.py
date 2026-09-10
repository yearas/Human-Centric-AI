from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .task2 import get_expert_predictions


def generate_deferral_plot(
    ai_acc, expert_acc, team_acc, beneficial_pct, harmful_pct, neutral_pct
):
    plots_dir = Path(settings.MEDIA_ROOT) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plots_dir / "task3_deferral_results.png"

    fig, (accuracy_ax, deferral_ax) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    label_style = {
        "ha": "center",
        "va": "bottom",
        "fontweight": "bold",
    }

    # Panel 1: Accuracy comparison
    accuracy_labels = ["AI Alone", "Expert Alone", "Team (AI+Expert)"]
    accuracy_values = [ai_acc * 100, expert_acc * 100, team_acc * 100]
    accuracy_bars = accuracy_ax.bar(accuracy_labels, accuracy_values, width=0.5)

    accuracy_ax.set_ylabel("Accuracy (%)", fontsize=11)
    accuracy_ax.set_title(
        "Overall Accuracy Comparison", fontsize=12, fontweight="bold"
    )
    accuracy_ax.set_ylim(0, 110)

    for bar in accuracy_bars:
        height = bar.get_height()
        center = bar.get_x() + bar.get_width() / 2
        label = f"{height:.1f}%"

        accuracy_ax.text(center, height + 2, label, **label_style)

    # Panel 2: Deferral quality
    deferral_labels = ["Beneficial", "Harmful", "Neutral"]
    deferral_values = [beneficial_pct, harmful_pct, neutral_pct]
    deferral_bars = deferral_ax.bar(deferral_labels, deferral_values, width=0.45)
    deferral_ax.set_ylabel("% of Deferred Cases", fontsize=11)
    deferral_ax.set_title(
        "Quality of Deferral Decisions", fontsize=12, fontweight="bold"
    )
    deferral_ax.set_ylim(0, max(deferral_values) + 4)

    for bar in deferral_bars:
        height = bar.get_height()
        center = bar.get_x() + bar.get_width() / 2
        label = f"{height:.2f}%"
        deferral_ax.text(center, height + 0.08, label, **label_style)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=130)
    plt.close()

    return f"{settings.MEDIA_URL}plots/task3_deferral_results.png"


def run_task3(
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
):
    # Run the existing Task 3 calculation and return its results.
    # Task 3 assumes expert predictions are available for all training examples.
    expert_train_preds, _ = get_expert_predictions(X_train, y_train, seed=1)
    y_train_arr = np.asarray(y_train)
    expert_correct_train = (expert_train_preds == y_train_arr).astype(int)

    # Reuse the Task 1 text representation to learn the expert's competence.
    tfidf = model.named_steps["tfidfvectorizer"]
    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    competence_model = LogisticRegression(max_iter=1000, random_state=42)
    competence_model.fit(X_train_tfidf, expert_correct_train)

    # Route using learned estimates only, never test labels or the simulator's
    # hidden expertise-topic mask.
    predicted_expert_accuracy = competence_model.predict_proba(X_test_tfidf)[:, 1]
    ai_conf = ai_probs.max(axis=1)
    defer = predicted_expert_accuracy > ai_conf

    # Use test labels only after routing to evaluate the completed system.
    team_preds = np.where(defer, expert_preds, y_pred)
    team_accuracy = round(float(accuracy_score(y_test_arr, team_preds)), 4)
    gain_over_ai = round(team_accuracy - accuracy, 4)
    deferral_rate = f"{float(np.mean(defer)) * 100:.2f}%"

    beneficial = defer & (y_pred != y_test_arr) & (expert_preds == y_test_arr)
    harmful = defer & (y_pred == y_test_arr) & (expert_preds != y_test_arr)
    neutral = defer & ~beneficial & ~harmful
    deferred_count = int(np.sum(defer))
    if deferred_count:
        beneficial_val = float(np.sum(beneficial)) / deferred_count * 100
        harmful_val = float(np.sum(harmful)) / deferred_count * 100
        neutral_val = float(np.sum(neutral)) / deferred_count * 100
    else:
        beneficial_val = 0.0
        harmful_val = 0.0
        neutral_val = 0.0
    beneficial_rate = f"{beneficial_val:.2f}%"
    harmful_rate = f"{harmful_val:.2f}%"
    neutral_rate = f"{neutral_val:.2f}%"

    deferral_plot_url = generate_deferral_plot(
        accuracy,
        expert_accuracy,
        team_accuracy,
        round(beneficial_val, 2),
        round(harmful_val, 2),
        round(neutral_val, 2),
    )

    return {
        "team_accuracy": team_accuracy,
        "gain_over_ai": gain_over_ai,
        "deferral_rate": deferral_rate,
        "beneficial_rate": beneficial_rate,
        "harmful_rate": harmful_rate,
        "neutral_rate": neutral_rate,
        "deferral_plot_url": deferral_plot_url,
    }
