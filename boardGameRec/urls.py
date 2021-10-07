from django.urls import path
from . import views
from .algorithms import matrix_factorization

app_name = 'boardGameRec'
urlpatterns = [
    path('test', views.test),
    path('testpost', views.test_post),
    path('update/gd', views.update_gd),
    path('update/gd/<int:user_id>', views.update_gd_one),
]
