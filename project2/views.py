from django.shortcuts import render

def index(request):
    return render(request, "project2/index.html")
