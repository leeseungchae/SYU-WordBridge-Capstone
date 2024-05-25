# Create your views here.
from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from .models import Translation
import googletrans


def register(request):
    if request.method == 'POST':
        # POST 요청으로부터 사용자 정보 가져오기
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # 사용자 이름과 이메일 중복 확인
        if User.objects.filter(username=username).exists():
            return JsonResponse({'message': 'Username already exists'}, status=409)
        if User.objects.filter(email=email).exists():
            return JsonResponse({'message': 'Email already exists'}, status=409)

        # 새로운 사용자 생성 및 로그인
        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        return JsonResponse({'message': 'User registered successfully'}, status=201)

    return render(request, 'register.html')


def logout_view(request):
    # 사용자 로그아웃
    logout(request)
    return JsonResponse({'message': 'Logout successful'}, status=204)


def login_view(request):
    if request.method == 'POST':
        # POST 요청으로부터 사용자 정보 가져오기
        username = request.POST.get('username')
        password = request.POST.get('password')

        # 사용자 인증
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # 인증 성공 시 로그인 및 성공 메시지 반환
            login(request, user)
            return JsonResponse({'message': 'Login successful'}, status=200)
        else:
            # 인증 실패 시 실패 메시지 반환
            return JsonResponse({'message': 'Invalid credentials'}, status=401)

    return render(request, 'login.html')


def check_email(request):
    if request.method == 'POST':
        # POST 요청으로부터 이메일 주소 가져오기
        email = request.POST.get('email')

        # 이메일 중복 확인
        if User.objects.filter(email=email).exists():
            return JsonResponse({'message': 'Email already in use'}, status=409)
        return JsonResponse({'message': 'Email available'}, status=200)


@login_required
def history(request):
    # 현재 사용자의 번역 기록 가져오기
    translations = Translation.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'history.html', {'translations': translations})


@login_required
def translate(request):
    if request.method == 'POST':
        # POST 요청으로부터 번역할 텍스트와 대상 언어 가져오기
        source_text = request.POST.get('source_text')
        target_language = request.POST.get('target_language')

        # Google Translate를 사용하여 번역 수행
        translator = googletrans.Translator()
        translated_text = translator.translate(source_text, dest=target_language).text

        # 번역 결과를 데이터베이스에 저장
        translation = Translation(
            user=request.user,
            source_text=source_text,
            translated_text=translated_text,
            target_language=target_language
        )
        translation.save()

        # 번역 결과를 JSON 형태로 반환
        return JsonResponse({'translated_text': translated_text}, status=200)

    return render(request, 'translate.html')