from django import forms

class LoginForm(forms.Form):
    username = forms.CharField(
        label='Username',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    password = forms.CharField(
        label='Password',
        max_length=100,
        widget=forms.PasswordInput(attrs={'class': 'form-control'})
    )

class TranslationForm(forms.Form):
    source_text = forms.CharField(
        label='Source Text',
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 5})
    )
    source_language = forms.ChoiceField(
        label='Source Language',
        choices=[
            ('ko', 'Korean'),
            ('en', 'English'),
            ('zh', 'Chinese'),
            ('ja', 'Japanese'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    target_language = forms.ChoiceField(
        label='Target Language',
        choices=[
            ('ko', 'Korean'),
            ('en', 'English'),
            ('zh', 'Chinese'),
            ('ja', 'Japanese'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )