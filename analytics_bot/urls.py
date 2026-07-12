from django.urls import path
from .views import chat,home


urlpatterns = [
    path('analytics/', chat, name='chat'),
    path('analytics', chat, name='chat_no_slash'),
    path('', home, name='ui'),
]
