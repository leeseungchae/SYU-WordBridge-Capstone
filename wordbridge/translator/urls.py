from django.urls import path
from django.views.generic import TemplateView
from .views import translate, login_page, history,register, login_view, check_email
app_name = 'translator'
urlpatterns = [
    path('', translate, name='translate'),
    path('login.html/', login_view, name='login'),
    path('login/', login_page, name='logout'),
    path('history/', history, name='history'),

]