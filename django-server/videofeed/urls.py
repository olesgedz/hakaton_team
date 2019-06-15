from django.urls import path
from django.conf.urls import url, include
from . import views

urlpatterns = [
    path('videoupload/', views.video_upload, name='videoupload'),
    path('feed/', views.livefe, name='livefe'),
]