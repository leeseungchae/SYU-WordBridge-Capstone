from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# 현재 시간 기반의 기본 username 설정
default_username = timezone.now().strftime('%Y%m%d%H%M%S')

# 유저 테이블, 번역 테이블 추가
class User(models.Model):
    id = models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
    password = models.CharField(max_length=100,default=None)
    username = models.CharField(max_length=50, default=default_username)

    def __str__(self):
        return self.username


class Translation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    source_text = models.TextField()
    translated_text = models.TextField()
    source_language = models.CharField(max_length=10)
    target_language = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.source_language} to {self.target_language}"
