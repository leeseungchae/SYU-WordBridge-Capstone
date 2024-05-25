from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User


# 유저 테이블, 번역 테이블 추가
class User(models.Model):
    name = models.CharField(max_length=100)
    userid = models.OneToOneField(User, on_delete=models.CASCADE)
    password = models.CharField(max_length=100)


    def __str__(self):
        return self.userid


class Translation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    source_text = models.TextField()
    translated_text = models.TextField()
    source_language = models.CharField(max_length=10)
    target_language = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.source_language} to {self.target_language}"
