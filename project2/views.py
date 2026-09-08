from django.shortcuts import render
from palmerpenguins import load_penguins
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def generate_counterfactuals(x_no_encoded, target_label, model, encoded_features, X_train, numerical_features, categorical_features, category_values, n_samples = 200, k = 5, max_attempts = 5):

    # MAD per encoded column from training data with fallback to 1 if MAD is 0 to avoid division by zero
    mad = (X_train - X_train.median()).abs().median()
    mad = mad.replace(0, 1)

    # Create a DataFrame with the same structure as the encoded features, initialized with zeros. This will be used to store the encoded version of the input data point.
    x_encoded = pd.get_dummies(pd.DataFrame([x_no_encoded]), columns=categorical_features).astype(float)
    x_encoded = x_encoded.reindex(columns=encoded_features, fill_value=0).iloc[0]

    n = n_samples
    noise_scale = 1.0

    # Attempt to generate counterfactuals up to max_attempts times, increasing the number of samples and noise scale if no valid counterfactuals are found
    for attempt in range(max_attempts):
        candidates = []
        for _ in range(n):
            perturbed = x_no_encoded.copy()
            for feature in numerical_features:
                # Calculate the standard deviation of the feature in the training data and scale it by 0.3 and the noise_scale to determine the amount of noise to add
                sigma = X_train[feature].std() * 0.3 * noise_scale
                perturbed[feature] = x_no_encoded[feature] + np.random.normal(0, sigma)
            for feature in categorical_features:
                if np.random.rand() < min(0.3 * noise_scale, 1.0):  # Adjust the probability of perturbing categorical features based on noise_scale
                    # Get all other possible values for this categorical feature
                    other_values = [val for val in category_values[feature] if val != x_no_encoded[feature]]
                    perturbed[feature] = np.random.choice(other_values)
            candidates.append(perturbed)

        candidates_df = pd.DataFrame(candidates)
        candidates_encoded = pd.get_dummies(candidates_df, columns=categorical_features).astype(float)
        candidates_encoded = candidates_encoded.reindex(columns=encoded_features, fill_value=0)

        # Use the trained model to predict the class labels of the perturbed candidates and create a mask to filter out those that match the target label
        predictions = model.predict(candidates_encoded)
        mask = predictions == target_label

        # If any valid counterfactuals are found i.e. candidates that match the target label, calculate their distances from the original data point and return the top k closest counterfactuals
        if mask.sum() > 0:
            valid = candidates_df[mask].reset_index(drop=True)
            valid_encoded = candidates_encoded[mask].reset_index(drop=True)
            distances = (valid_encoded.sub(x_encoded, axis=1).abs() / mad).sum(axis=1)
            valid['distance'] = distances.round(3)
            return valid.sort_values(by='distance').head(k)

        # If no valid counterfactuals are found, increase the number of samples and noise scale for the next attempt
        n = int(n * 1.5)
        noise_scale *= 1.5

    # If no valid counterfactuals are found after max_attempts, return None
    return None


def compute_pdp(model, X, feature, grid):
    pdp_values = []
    for value in grid:
        X_temp = X.copy()
        X_temp[feature] = value
        # Use the trained model to predict the probabilities of each class for the modified dataset and calculate the mean predicted probabilities across all samples for this feature value
        predictions = model.predict_proba(X_temp)
        pdp_values.append(predictions.mean(axis=0))
    return np.array(pdp_values)


def compute_ale(model, X, feature, n_bins=10):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(X[feature].quantile(quantiles).values)
    n_bin_actual = len(bin_edges) - 1

    bin_indices = pd.cut(X[feature], bins=bin_edges, labels=False, include_lowest=True)

    n_classes = len(model.classes_)
    local_effects = np.zeros((n_bin_actual, n_classes))
    counts = np.zeros(n_bin_actual)

    for k in range(n_bin_actual):
        bin_mask = (bin_indices == k).values
        counts[k] = bin_mask.sum()
        if counts[k] == 0:
            continue
        X_lower = X[bin_mask].copy()
        X_upper = X[bin_mask].copy()
        X_lower[feature] = bin_edges[k]
        X_upper[feature] = bin_edges[k + 1]
        local_effects[k] = (model.predict_proba(X_upper) - model.predict_proba(X_lower)).mean(axis=0)

    ale_uncentered = np.cumsum(local_effects, axis=0)
    weights = counts / counts.sum()
    ale_centered = ale_uncentered - (ale_uncentered * weights[:, None]).sum(axis=0)

    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_midpoints, ale_centered

def index(request):
    # Load the penguins dataset and drop rows with missing values (11 rows)
    df = load_penguins().dropna()

    X = df.drop('species', axis=1)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    y = df['species']
    # Split the dataset into training and testing sets with stratification, ensuring that the class distribution is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Get the model type from the GET request, default to 'decision_tree'
    model_type = request.GET.get('model_type', 'decision_tree')
    # Get the regularization parameter 'lambda_' from the GET request. If it's not provided, default to 0 (no regularization).
    lambda_ = float(request.GET.get('lambda_', 0))

    model_variations = []

    if model_type == 'decision_tree':
        model_unregularized = DecisionTreeClassifier(random_state=42)
        model_unregularized.fit(X_train, y_train)
        # Get the maximum number of leaves in the unregularized decision tree model. 
        # This value can be used to understand the complexity of the model and to compare it with regularized models that may have fewer leaves due to pruning or other regularization techniques.
        max_leaves = model_unregularized.get_n_leaves()

        # Train decision tree models with different numbers of leaves (from 2 to max_leaves) and store their accuracy and number of leaves in a list
        for i in range(2, max_leaves + 1):
            model_trees = DecisionTreeClassifier(max_leaf_nodes=i, random_state=42)
            model_trees.fit(X_train, y_train)
            model_variations.append({'model': model_trees, 'accuracy': accuracy_score(y_test, model_trees.predict(X_test)), 'complexity_measure': model_trees.get_n_leaves()})

    else: # If the model type is not 'decision_tree', use a logistic regression model with standardization and regularization.
        for C in np.logspace(-3, 2, 15):
            model_logistic = Pipeline([
                ('scaler', StandardScaler()),
                ('logistic', LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=3000, random_state=42))
            ])
            model_logistic.fit(X_train, y_train)
            coefficients = model_logistic.named_steps['logistic'].coef_
            # Count the number of non-zero coefficients in the logistic regression model. This serves as a measure of model complexity
            non_zero_coefficients = np.sum(np.any(coefficients != 0, axis=0))
            model_variations.append({'model': model_logistic, 'accuracy': accuracy_score(y_test, model_logistic.predict(X_test)), 'complexity_measure': non_zero_coefficients})

    # Select the best model based on the trade-off between accuracy and complexity, using the provided lambda_ parameter to weight the importance of complexity in the selection process.
    best = max(model_variations, key=lambda x: x['accuracy'] - lambda_ * x['complexity_measure'])
    model = best['model']

    accuracy = round(best['accuracy'], 3)
    n_leaves = best['complexity_measure']


    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)

    # Create a confusion matrix and convert it to an HTML table for display in the template
    # labels=model.classes_ ensures that the confusion matrix has the correct order of classes
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_matrix_table = pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_).to_html()

    original_features = df.drop('species', axis=1).reset_index(drop=True)
    numerical_features = original_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = original_features.select_dtypes(include=['object', 'category']).columns.tolist()
    category_values = {feature: original_features[feature].unique().tolist() for feature in categorical_features}
    available_indices = original_features.index.tolist()
    available_labels = sorted(y.unique())

    x_index = int(request.GET.get('x_index', 0))
    target_label = request.GET.get('target_label', sorted(y.unique())[0])  # Default to the first class if not provided
    x_no_encoded = original_features.iloc[x_index]

    counterfactuals = generate_counterfactuals(x_no_encoded, target_label, model, X.columns, X_train, numerical_features, categorical_features, category_values)

    pdp_ale_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    selected_feature = request.GET.get('feature', pdp_ale_features[0])  # Default to the first numerical feature if not provided

    grid = np.linspace(X[selected_feature].min(), X[selected_feature].max(), 20)
    pdp_values = compute_pdp(model, X, selected_feature, grid)
    bin_midpoints, ale_values = compute_ale(model, X, selected_feature)


    context = {
        "accuracy": accuracy,
        "complexity_measure": n_leaves,
        "lambda_": lambda_,
        "report": report,
        "conf_matrix_table": conf_matrix_table,
        "model_type": model_type,
        "available_indices": available_indices,
        "available_labels": available_labels,
        "x_index": x_index,
        "target_label": target_label,
        "selected_feature": selected_feature,
        "available_features": pdp_ale_features,
        "counterfactuals_table": counterfactuals.to_html() if counterfactuals is not None else None,
    }

    if model_type == 'decision_tree':
        # Figsize is set to (24, 14) to ensure that the decision tree plot is large enough to be easily readable, especially when there are many features or classes. 
        # The fontsize is set to 10 for better readability of the text in the plot
        plt.figure(figsize=(24, 14))
        plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, fontsize=10)
        plt.title(f'Decision Tree Classifier (lambda={lambda_}, complexity_measure={n_leaves})', fontsize=16)
        plt.tight_layout()
        plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_filename = 'plots/decision_tree_project2.png'
        plt.savefig(os.path.join(settings.MEDIA_ROOT, plot_filename), dpi=150, bbox_inches='tight')
        plt.close()
        context['image_url'] = settings.MEDIA_URL + plot_filename
    else:
        coefficients_df = pd.DataFrame(model.named_steps['logistic'].coef_, index=model.named_steps['logistic'].classes_, columns=X.columns,).round(3)
        context['coefficients_table'] = coefficients_df.to_html()


    plt.figure(figsize=(10, 6))
    for i, class_label in enumerate(model.classes_):
        plt.plot(grid, pdp_values[:, i], label=class_label)
    plt.xlabel(selected_feature)
    plt.ylabel('Predicted probability')
    plt.title(f'PDP: {selected_feature}')
    plt.legend()
    pdp_filename = 'plots/pdp_project2.png'
    plt.savefig(os.path.join(settings.MEDIA_ROOT, pdp_filename), dpi=150, bbox_inches='tight')
    plt.close()
    context['pdp_image_url'] = settings.MEDIA_URL + pdp_filename

    plt.figure(figsize=(10, 6))
    for i, class_label in enumerate(model.classes_):
        plt.plot(bin_midpoints, ale_values[:, i], label=class_label)
    plt.xlabel(selected_feature)
    plt.ylabel('ALE')
    plt.title(f'ALE: {selected_feature}')
    plt.legend()
    ale_filename = 'plots/ale_project2.png'
    plt.savefig(os.path.join(settings.MEDIA_ROOT, ale_filename), dpi=150, bbox_inches='tight')
    plt.close()
    context['ale_image_url'] = settings.MEDIA_URL + ale_filename


    return render(request, "project2/index.html", context)