from pathlib import Path
import joblib
from django.conf import settings
from django.shortcuts import render
from datasets import load_dataset

from .task1 import run_task1
from .task2 import run_task2

CACHE_FILE = Path(__file__).resolve().parent / "cache.joblib"
MODEL_FILE = Path(__file__).resolve().parent / "saved_models/baseline_model.joblib"
CACHE_VERSION = 1


def index(request):
    # Fast load: return cached results if already computed
    if CACHE_FILE.exists():
        data = joblib.load(CACHE_FILE)
        plot_path = Path(settings.MEDIA_ROOT) / "plots/expert_accuracy_task2.png"
        if data.get("cache_version") == CACHE_VERSION and plot_path.exists():
            return render(request, "project3/index.html", data)

    # 1. Load AG News dataset
    dataset = load_dataset("fancyzhx/ag_news")
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]
    X_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    # 2. Task 1: Baseline AI Model
    task1_results = run_task1(X_train, y_train, X_test, y_test, MODEL_FILE)

    # 3. Task 2: Simulated Expert
    task2_results = run_task2(X_test, y_test)

    context = {
        "cache_version": CACHE_VERSION,
        # Task 1
        "accuracy": task1_results["accuracy"],
        "report": task1_results["report"],
        "conf_matrix_table": task1_results["conf_matrix_table"],
        "complexity_measure": task1_results["complexity_measure"],
        # Task 2
        "expert_accuracy": task2_results["expert_accuracy"],
        "expert_inside_acc": task2_results["expert_inside_acc"],
        "expert_outside_acc": task2_results["expert_outside_acc"],
        "expert_topic_coverage": task2_results["expert_topic_coverage"],
        "expert_region_coverage": task2_results["expert_topic_coverage"],
        "expert_plot_url": task2_results["expert_plot_url"],
    }

    # Save cache for instant future loads
    joblib.dump(context, CACHE_FILE)

    return render(request, "project3/index.html", context)
