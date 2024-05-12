from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser): # 장고에서 제공하는 기본 사용자 모델, 기본적인 정보가 이미 있음
    pass

class TranslationHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='translation_histories')
    source_text = models.TextField() # 길이 제한 없음
    source_language = models.CharField(max_length=10)
    target_language = models.CharField(max_length=10)
    translated_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)