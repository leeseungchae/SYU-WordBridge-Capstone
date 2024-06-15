# Create your views here.
from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import Translation, User
import googletrans
from django.views.decorators.csrf import csrf_exempt
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
en_2_ko_tokenizer = AutoTokenizer.from_pretrained("NHNDQ/nllb-finetuned-en2ko")
en_2_ko_model = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-en2ko").to(device)

ko_2_en_tokenizer = AutoTokenizer.from_pretrained("NHNDQ/nllb-finetuned-ko2en")
ko_2_en_model= AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-ko2en").to(device)
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
        # userId = json_data['userId']
        if 'username' in json_data and 'password' in json_data:
            userId = json_data['username']
            password = json_data['password']

        user = authenticate(request, username=userId, password=password)
        if user is not None:
            # 인증 성공 시 로그인 및 성공 메시지 반환
            login(request, user)
            return JsonResponse({'message': 'Login successful'}, status=200)
        else:
            # 인증 실패 시 실패 메시지 반환
            return JsonResponse({'message': 'Invalid credentials'}, status=401)

    #return JsonResponse({'message': 'Invalid credentials'}, status=401)
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


# @login_required
# def translate(request):
#     if request.method == 'POST':
#         # POST 요청으로부터 번역할 텍스트와 대상 언어 가져오기
#         source_text = request.POST.get('source_text')
#         target_language = request.POST.get('target_language')
#
#         # Google Translate를 사용하여 번역 수행
#         translator = googletrans.Translator()
#         translated_text = translator.translate(source_text, dest=target_language).text
#
#         # 번역 결과를 데이터베이스에 저장
#         translation = Translation(
#             user=request.user,
#             source_text=source_text,
#             translated_text=translated_text,
#             target_language=target_language
#         )
#         translation.save()
#
#         # 번역 결과를 JSON 형태로 반환
#         return JsonResponse({'translated_text': translated_text}, status=200)
#
#     return render(request, 'translate.html')
#
#
# @csrf_exempt
# def translate(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         text = data.get('q')
#         target_language = data.get('target')
#         # POST 요청으로부터 번역할 텍스트와 대상 언어 가져오기
#         # source_text = request.POST.get('source_text')
#         # target_language = request.POST.get('target_language')
#
#         print(target_language)
#
#         if target_language == 'ko':
#             tokenizer = en_2_ko_tokenizer
#             model = en_2_ko_model
#         elif target_language == 'en':
#             tokenizer = ko_2_en_tokenizer
#             model = ko_2_en_model
#
#         # 입력 텍스트를 토큰화합니다.
#         inputs = tokenizer(text, return_tensors="pt").to(model.device)
#
#         # 모델을 사용하여 추론합니다.
#         with torch.no_grad():
#             outputs = model.generate(**inputs)
#         # 생성된 토큰을 디코딩하여 한국어 텍스트로 변환합니다.
#         translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#         print(translated_text)
#
#         return translated_text

@csrf_exempt
def translate(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('q')
            target_language = data.get('target')

            if not text or not target_language:
                return JsonResponse({'error': 'Invalid input'}, status=400)

            if target_language == 'ko':
                tokenizer = en_2_ko_tokenizer
                model = en_2_ko_model
            elif target_language == 'en':
                tokenizer = ko_2_en_tokenizer
                model = ko_2_en_model
            else:
                return JsonResponse({'error': 'Unsupported target language'}, status=400)

            # 입력 텍스트를 토큰화합니다.
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # 모델을 사용하여 추론합니다.
            with torch.no_grad():
                outputs = model.generate(**inputs)

            # 생성된 토큰을 디코딩하여 한국어 텍스트로 변환합니다.
            translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print(translated_text)
            return JsonResponse({'translated_text': translated_text}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'translate.html')