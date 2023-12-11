"""
URL configuration for app_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from backend import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('upload/', views.upload_photo, name='upload_photo'),
    path('login/', views.login, name='login'),
    # path('uploadvideo/', views.upload_video, name='upload_video'),
    path('uploadlocation/', views.upload_location, name='upload_location'),
    path('uploadid/', views.upload_ID, name='upload_id'),
    path('uploadImageForSeminar/', views.upload_photo_for_seminar, name='upload_photo_for_seminar'),
    path('update/', views.update, name='update'),
]
