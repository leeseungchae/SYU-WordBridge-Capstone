from django.urls import path
from django.views.generic import TemplateView
from . import views

urlpatterns = [
    path('', views.translate, name='translate'),
    path('login.html/', views.login_view, name='login'),
    path('login/', TemplateView.as_view(template_name='login.html'), name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('check-email/', views.check_email, name='check_email'),
    path('history/', views.history, name='history'),
    path('register/', views.register, name='register'),
]