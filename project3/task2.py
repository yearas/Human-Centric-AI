from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings


# Define the sports keywords used to identify articles in the expert's specialty area
SPORTS_WORDS = frozenset(
    "football soccer basketball baseball tennis cricket rugby hockey "
    "olympic olympics championship tournament league match goal goals "
    "coach striker playoffs athlete athletes medal medals nba nfl "
    "fifa cup racing grandprix pitcher inning innings touchdown".split()
)


def in_expertise_topic(text):
    # Convert the article to lowercase words and check if at least two sports keywords are present
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & SPORTS_WORDS) >= 2


in_expertise_region = in_expertise_topic


def get_expert_predictions(texts, labels, seed=42):
    rng = np.random.RandomState(seed)
    expert_preds = []
    in_topic_mask = []

    for text, label in zip(texts, labels):
        is_expert = in_expertise_topic(text)
        in_topic_mask.append(is_expert)

        # Set the probability of a correct answer to 0.95 inside the sports topic and 0.55 outside it
        prob = 0.95 if is_expert else 0.55
        if rng.rand() < prob:
            expert_preds.append(int(label))
        else:
            # Choose one of the other classes when the expert gives an incorrect answer
            wrong = [c for c in (0, 1, 2, 3) if c != label]
            expert_preds.append(int(rng.choice(wrong)))

    return np.array(expert_preds), np.array(in_topic_mask)


def generate_expert_plot(inside_acc, outside_acc, overall_acc):
    plots_dir = Path(settings.MEDIA_ROOT) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plots_dir / "expert_accuracy_task2.png"

    categories = ["Inside Specialty\n(Sports)", "Outside Specialty\n(Non-Sports)", "Overall Expert"]
    values = [inside_acc * 100, outside_acc * 100, overall_acc * 100]

    # Create a bar chart to compare the expert's accuracy inside and outside the sports topic and overall
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(categories, values, width=0.5)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Simulated Expert Accuracy: Strengths vs. Weaknesses", fontsize=13)
    ax.set_ylim(0, 110)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(plot_file, dpi=130)
    plt.close(fig)

    return f"{settings.MEDIA_URL}plots/expert_accuracy_task2.png"


def run_task2(X_test, y_test):
    # Get the simulated expert's predictions for the test articles
    expert_preds, in_topic = get_expert_predictions(X_test, y_test)
    y_test_arr = np.array(y_test)

    # Calculate the overall accuracy and the accuracy inside and outside the sports topic
    expert_accuracy = round(float(np.mean(expert_preds == y_test_arr)), 4)
    expert_inside_acc = round(
        float(np.mean(expert_preds[in_topic] == y_test_arr[in_topic])), 4
    )
    expert_outside_acc = round(
        float(np.mean(expert_preds[~in_topic] == y_test_arr[~in_topic])), 4
    )
    expert_topic_coverage = f"{round(float(np.mean(in_topic)) * 100, 2)}%"
    expert_plot_url = generate_expert_plot(
        expert_inside_acc, expert_outside_acc, expert_accuracy
    )

    return {
        "expert_preds": expert_preds,
        "in_topic": in_topic,
        "in_region": in_topic,
        "y_test_arr": y_test_arr,
        "expert_accuracy": expert_accuracy,
        "expert_inside_acc": expert_inside_acc,
        "expert_outside_acc": expert_outside_acc,
        "expert_topic_coverage": expert_topic_coverage,
        "expert_region_coverage": expert_topic_coverage,
        "expert_plot_url": expert_plot_url,
    }
