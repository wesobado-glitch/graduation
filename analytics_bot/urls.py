from django.urls import path
from .views import chat


urlpatterns = [
    path('analytics/', chat, name='chat'),
    path('analytics', chat, name='chat_no_slash'),
]