import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import UploadFileForm, TrainingForm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASSIFICATION_THRESHOLD = 20  # Threshold for determining classification vs regression
def is_classification_target(series):
    return not pd.api.types.is_numeric_dtype(series) or series.nunique() <= CLASSIFICATION_THRESHOLD



# Get the model and hyperparameter grid for a given model type
# 3 model types are supported: 'knn', 'decision_tree', and 'ridge' for regression/classification tasks.
# The function returns a model constructor and a hyperparameter grid based on the provided min and max values.
def get_model_and_grid(is_classification, model_type, hyperparam_min, hyperparam_max, num_steps=10):
    if model_type == 'knn':
        grid = list(range(max(1, int(hyperparam_min)), int(hyperparam_max) + 1))
        if is_classification:
            model = lambda v: KNeighborsClassifier(n_neighbors=v)
        else:
            model = lambda v: KNeighborsRegressor(n_neighbors=v)
    elif model_type == 'decision_tree':
        grid = list(range(max(1, int(hyperparam_min)), int(hyperparam_max) + 1))
        if is_classification:
            model = lambda v: DecisionTreeClassifier(max_depth=v, random_state=42)
        else:
            model = lambda v: DecisionTreeRegressor(max_depth=v, random_state=42)
    elif model_type == 'ridge':
        grid = list(np.logspace(np.log10(hyperparam_min), np.log10(hyperparam_max), num_steps))
        if is_classification:
            model = lambda v: RidgeClassifier(alpha=v)
        else:
            model = lambda v: Ridge(alpha=v)
    return model, grid

# Compute evaluation metrics based on the type of model (classification or regression)
def compute_metrics(y_true, y_pred, is_classification):
    if is_classification:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
    else:
        return {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
        }


def index(request):
    return render(request, 'project1/index.html')

def visualization(request):
    filename = request.session.get('csv_filename')

    # If no filename is found in the session, redirect to the upload page
    if not filename:
        return redirect('project1:upload_csv')

    # Read the CSV file into a DataFrame
    csv_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
    df = pd.read_csv(csv_path)

    # Determine feature and target columns
    feature_columns = df.columns[:-1]
    target_column = df.columns[-1]

    # Get selected features from GET parameters, defaulting to the first two feature columns
    feature_x = request.GET.get('feature_x', feature_columns[0])
    feature_y = request.GET.get('feature_y', feature_columns[1])

    if feature_x not in feature_columns:
        feature_x = feature_columns[0]
    if feature_y not in feature_columns:
        feature_y = feature_columns[1]


    plt.figure()
    if is_classification_target(df[target_column]):
        for label, group in df.groupby(target_column):
            plt.scatter(group[feature_x], group[feature_y], label=str(label))
        plt.legend()
    else:
        plt.scatter(df[feature_x], df[feature_y])
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)

    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = f'plots/{request.session.session_key}_{feature_x}_{feature_y}.png'
    plt.savefig(os.path.join(settings.MEDIA_ROOT, plot_filename))
    plt.close()

    return render(request, 'project1/visualization.html', {
        'image_url': settings.MEDIA_URL + plot_filename,
        'feature_x': feature_x,
        'feature_y': feature_y,
        'feature_columns': feature_columns,
    })


def upload_csv(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        # Check if the form is valid
        if form.is_valid():
            df = pd.read_csv(request.FILES['file'])
            # Save the DataFrame to a CSV file in the media/uploads directory
            if not request.session.session_key:
                request.session.save()
            session_key = request.session.session_key
            # Create the uploads directory if it doesn't exist
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            # Save the DataFrame to a CSV file named with the session key
            save_path = os.path.join(upload_dir, f'{session_key}.csv')
            df.to_csv(save_path, index=False)
            # Store the filename in the session for later use
            request.session['csv_filename'] = f'{session_key}.csv'
            return redirect('project1:visualization')
    else:
        form = UploadFileForm()
    # Render the upload form template with the form context
    return render(request, 'project1/upload.html', {'form': form})


def training(request):
    filename = request.session.get('csv_filename')
    if not filename:
        return redirect('project1:upload_csv')

    if request.method == 'POST':
        form = TrainingForm(request.POST)
        if form.is_valid():
            csv_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
            df = pd.read_csv(csv_path)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            is_classification = is_classification_target(y)

            data = form.cleaned_data
            model, grid = get_model_and_grid(
                is_classification, data['model_type'], data['hyperparameter_min'], data['hyperparameter_max']
            )

            results = []
            if data['evaluation_method'] == 'split':
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=data['test_size'], random_state=42
                )
                for value in grid:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model(value)),
                    ])
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    metrics = compute_metrics(y_test, y_pred, is_classification)
                    # Append the results with hyperparameter value, model class name, and computed metrics
                    results.append({'hyperparameter': value, **metrics})
            elif data['evaluation_method'] == 'cv':
                
                for value in grid:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model(value)),
                    ])
                    scores = cross_val_score(pipeline, X, y, cv=data['cv_folds'], scoring='accuracy' if is_classification else 'r2')
                    # Append the results with hyperparameter value, model class name, and computed metrics
                    results.append({'hyperparameter': value,'mean score': np.mean(scores), 'std score': np.std(scores)})

            return render(request, 'project1/training.html', {'form': form, 'results': results})

    else:
        form = TrainingForm()
    return render(request, 'project1/training.html', {'form': form})

