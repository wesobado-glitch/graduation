from django.urls import path
from .views import recommend, recommend_by_customer

urlpatterns = [
    path('recommend/', recommend),
    path('recommend', recommend),
    path('recommend/customer/', recommend_by_customer),
    path('recommend/customer', recommend_by_customer),
]

