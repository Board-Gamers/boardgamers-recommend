from django.urls import path
from . import views
from .algorithms import matrix_factorization

app_name = 'boardGameRec'
urlpatterns = [
    path('update/gd', views.update_gd),
]
