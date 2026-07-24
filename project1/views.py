import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from .forms import UploadFileForm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    # Use the first two feature columns for visualization
    feature_x, feature_y = feature_columns[0], feature_columns[1]

    # Flag for classification, heuristically determined based on the target column's data type and unique values
    is_classification = (
        not pd.api.types.is_numeric_dtype(df[target_column])
        or df[target_column].nunique() <= 20
    )

    plt.figure()
    if is_classification:
        for label, group in df.groupby(target_column):
            plt.scatter(group[feature_x], group[feature_y], label=str(label))
        plt.legend()
    else:
        plt.scatter(df[feature_x], df[feature_y])
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)

    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = f'plots/{request.session.session_key}.png'
    plt.savefig(os.path.join(settings.MEDIA_ROOT, plot_filename))
    plt.close()

    return render(request, 'project1/visualization.html', {
        'image_url': settings.MEDIA_URL + plot_filename,
        'feature_x': feature_x,
        'feature_y': feature_y,
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
