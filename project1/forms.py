from django import forms


class UploadFileForm(forms.Form):
    file = forms.FileField(label='Select a (csv) file')

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if not file.name.endswith('.csv'):
                raise forms.ValidationError("Only csv files are allowed!")
        return file


class TrainingForm(forms.Form):
    MODEL_SELECTION = [
        ('ridge', 'Ridge'),
        ('decision_tree', 'Decision Tree'),
        ('knn', 'k-Nearest Neighbors'),
    ]
    EVALUATION_CHOICES = [
        ('split', 'Train-Test Split'),
        ('cv', 'Cross-Validation'),
    ]

    model_type = forms.ChoiceField(choices=MODEL_SELECTION, label='Model')
    evaluation_method = forms.ChoiceField(choices=EVALUATION_CHOICES, label='Evaluation method')
    test_size = forms.FloatField(label='Test set fraction', min_value=0.05, max_value=0.5, initial=0.2, required=False)
    cv_folds = forms.IntegerField(label='Number of CV folds', min_value=2, max_value=10, initial=5, required=False)
    hyperparameter_min = forms.FloatField(label='Hyperparameter min')
    hyperparameter_max = forms.FloatField(label='Hyperparameter max')

    def clean(self):
        cleaned_data = super().clean()
        model_type = cleaned_data.get('model_type')
        hmin = cleaned_data.get('hyperparameter_min')
        hmax = cleaned_data.get('hyperparameter_max')

        if hmin is not None and hmax is not None:
            if hmin >= hmax:
                self.add_error('hyperparameter_max', 'Must be greater than hyperparameter min.')
            if model_type == 'ridge' and hmin <= 0:
                self.add_error('hyperparameter_min', 'For Ridge, must be greater than 0.')
            if model_type == 'decision_tree' and hmin <= 0:
                self.add_error('hyperparameter_min', 'For Decision Tree, must be greater than 0.')
            if model_type == 'knn' and hmin <= 0:
                self.add_error('hyperparameter_min', 'For k-Nearest Neighbors, must be greater than 0.')

        return cleaned_data
