from pathlib import Path
import joblib
from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render
from datasets import load_dataset

from .task1 import run_task1
from .task2 import run_task2
from .task3 import run_task3
from .task4 import run_task4

CACHE_FILE = Path(__file__).resolve().parent / "cache.joblib"
MODEL_FILE = Path(__file__).resolve().parent / "saved_models/baseline_model.joblib"
CACHE_VERSION = 7


def index(request):
    # Fast load: return cached results if already computed
    if CACHE_FILE.exists():
        data = joblib.load(CACHE_FILE)
        plot_files = [
            Path(settings.MEDIA_ROOT) / "plots/expert_accuracy_task2.png",
            Path(settings.MEDIA_ROOT) / "plots/task3_deferral_results.png",
            Path(settings.MEDIA_ROOT) / "plots/task4_learning_curve.png",
        ]
        if data.get("cache_version") == CACHE_VERSION and all(
            plot_file.exists() for plot_file in plot_files
        ):
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

    # 4. Task 3: Learning to Defer
    task3_results = run_task3(
        X_train,
        y_train,
        X_test,
        task2_results["y_test_arr"],
        task1_results["model"],
        task1_results["ai_probs"],
        task1_results["y_pred"],
        task1_results["accuracy"],
        task2_results["expert_preds"],
        task2_results["expert_accuracy"],
    )

    # 5. Task 4: Active Learning for Expert Competence Discovery
    task4_results = run_task4(
        X_train,
        y_train,
        X_test,
        task2_results["y_test_arr"],
        task1_results["model"],
        task1_results["ai_probs"],
        task1_results["y_pred"],
        task1_results["accuracy"],
        task2_results["expert_preds"],
        task2_results["expert_accuracy"],
    )

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
        "expert_topic_count": task2_results["expert_topic_count"],
        "expert_topic_coverage": task2_results["expert_topic_coverage"],
        "expert_region_coverage": task2_results["expert_topic_coverage"],
        "expert_plot_url": task2_results["expert_plot_url"],
        # Task 3
        "team_accuracy": task3_results["team_accuracy"],
        "gain_over_ai": task3_results["gain_over_ai"],
        "deferral_rate": task3_results["deferral_rate"],
        "beneficial_rate": task3_results["beneficial_rate"],
        "harmful_rate": task3_results["harmful_rate"],
        "neutral_rate": task3_results["neutral_rate"],
        "deferral_plot_url": task3_results["deferral_plot_url"],
        # Task 4
        "budgets": task4_results["budgets"],
        "task4_rows": task4_results["task4_rows"],
        "learning_curve_plot_url": task4_results["learning_curve_plot_url"],
    }

    # Save cache for instant future loads
    joblib.dump(context, CACHE_FILE)

    return render(request, "project3/index.html", context)


def download_report(request):
    report_path = Path(__file__).resolve().parent / "Report.pdf"

    try:
        report_file = report_path.open("rb")
    except FileNotFoundError:
        raise Http404("The report is not available.")

    return FileResponse(
        report_file,
        as_attachment=True,
        filename="Project3_Report.pdf",
        content_type="application/pdf",
    )
