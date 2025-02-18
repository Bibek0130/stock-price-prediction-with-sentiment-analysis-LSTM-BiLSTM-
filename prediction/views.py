from django.shortcuts import render, redirect
import joblib
from django.http import HttpResponseBadRequest

#rendering about page
def about(request):
    return render(request, 'prediction/about.html')