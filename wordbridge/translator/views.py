# Create your views here.
from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import Translation, User
import googletrans
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def register(request):
    if request.method == 'POST':
        json_data = json.loads(request.body)
     # 사용자 이름과 이메일 중복 확인
        user = User.objects.create(username=json_data['userId'], password=json_data['password'])
        user.save()

        return JsonResponse({'message': 'User registered successfully'}, status=201)

    return render(request, 'register.html')


def login_page(request):

    return render(request, 'login.html')
@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        json_data = json.loads(request.body)
        if json_data.get('userId')is None:
            return JsonResponse({'message': 'Invalid credentials'}, status=401)
        user = authenticate(request, username=json_data['userId'], password=json_data['password'])
        print(user)
        if user is not None:
            # 인증 성공 시 로그인 및 성공 메시지 반환
            login(request, user)
            return JsonResponse({'message': 'Login successful'}, status=200)
        else:
            # 인증 실패 시 실패 메시지 반환
            return JsonResponse({'message': 'Invalid credentials'}, status=401)

    return render(request, 'login.html')

@csrf_exempt
def check_email(request, userId):
    if request.method == 'GET':
        if User.objects.filter(username=userId).exists():
            return JsonResponse({'message': 'Email already in use'}, status=409)
        else:
            return JsonResponse({'message': 'Email available'}, status=200)
    else:
        return JsonResponse({'message': 'Invalid credentials'}, status=401)

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