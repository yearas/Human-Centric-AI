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



def index(request):
    # Load the penguins dataset and drop rows with missing values (11 rows)
    df = load_penguins().dropna()

    X = df.drop('species', axis=1)
    X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
    y = df['species']
    # Split the dataset into training and testing sets with stratification, ensuring that the class distribution is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)

    # Create a confusion matrix and convert it to an HTML table for display in the template
    # labels=model.classes_ ensures that the confusion matrix has the correct order of classes
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_matrix_table = pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_).to_html()


    accuracy = round(accuracy_score(y_test, y_pred), 3)

    n_leaves = model.get_n_leaves()

    # Figsize is set to (24, 14) to ensure that the decision tree plot is large enough to be easily readable, especially when there are many features or classes. 
    # The fontsize is set to 10 for better readability of the text in the plot.
    plt.figure(figsize=(24, 14))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, fontsize=10)
    plt.title(f'Decision Tree Classifier (n_leaves={n_leaves})')
    plt.xlabel('Features')
    plt.ylabel('Class')
    plt.tight_layout()
    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = 'plots/decision_tree_project2.png'
    plt.savefig(os.path.join(settings.MEDIA_ROOT, plot_filename), dpi=150, bbox_inches='tight')
    plt.close()

    return render(request, "project2/index.html", {"accuracy": accuracy, "n_leaves": n_leaves, "image_url": settings.MEDIA_URL + plot_filename, "report": report, "conf_matrix_table": conf_matrix_table})
