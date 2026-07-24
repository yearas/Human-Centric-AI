from django import forms


class UploadFileForm(forms.Form):
    file = forms.FileField(label='Select a (csv) file')

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if not file.name.endswith('.csv'):
                raise forms.ValidationError("Only csv files are allowed!")
        return file