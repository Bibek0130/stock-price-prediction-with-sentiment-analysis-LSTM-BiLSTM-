from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('predict/',views.predict, name='predict'),
    path('about/',views.about, name='about'),
]
