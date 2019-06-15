from django import forms
from django.core.validators import FileExtensionValidator

class UploadVideoFileForm(forms.Form):
    video = forms.FileField()