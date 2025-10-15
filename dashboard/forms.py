from django.shortcuts import render

# Create your views here.
from django import forms

class UploadForm(forms.Form):
    file = forms.FileField(help_text="Upload Cortical_waveforms.xlsx")

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith(".xlsx"):
            raise forms.ValidationError("Please upload a .xlsx file.")
        return f
