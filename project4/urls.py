from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path("", views.index, name="index"),
    path("start/", views.start_study, name="start_study"),
    path("trial/", views.trial, name="trial"),
    path("result/", views.result, name="result"),
]