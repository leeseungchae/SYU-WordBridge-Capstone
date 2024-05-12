from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import LoginForm, TranslationForm
from .models import TranslationHistory


translator = Translator()

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('translation')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def translation_view(request):
    translation_histories = TranslationHistory.objects.filter(user=request.user)
    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            source_text = form.cleaned_data['source_text']
            source_language = form.cleaned_data['source_language']
            target_language = form.cleaned_data['target_language']

            # 번역 수행
            translated_text = translator.translate(source_text, src=source_language, dest=target_language).text

            # 번역 기록 저장
            TranslationHistory.objects.create(
                user=request.user,
                source_text=source_text,
                source_language=source_language,
                target_language=target_language,
                #translated_text=translated_text
            )
    else:
        form = TranslationForm()

    return render(request, 'translation.html', {
        'form': form,
        'translation_histories': translation_histories
    })