import time
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from . import functions
import cv2
import numpy as np
from PIL import Image
from django.contrib.auth import authenticate
import json
from .functions import test_combine
from .functions import only_for_seminar
import os
import base64


@csrf_exempt
def login(request):
    if request.method == "POST":
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")
        valid = authenticate(request, username=username, password=password)
        if valid is not None:
            return JsonResponse({"message": "success"})
        else:
            return JsonResponse({"message": "wrong username or password"}, status=401)
    return JsonResponse({"message": "POST request only"}, status=400)


### app for seminar
@csrf_exempt
def upload_location(request):
    if request.method == "POST":
        data = json.loads(request.body)
        latitude = data.get("Latitude")
        longitude = data.get("Longitude")
        images_to_return = only_for_seminar.find_near_location(latitude, longitude)
        response_data = {"success": True, "images": images_to_return}
        return JsonResponse(response_data)
    else:
        return JsonResponse({"success": False})


@csrf_exempt
def upload_ID(request):
    if request.method == "POST":
        data = json.loads(request.body)
        imageid = data.get("ImageID")
        images_to_return = only_for_seminar.find_image(imageid)
        response_data = {"success": True, "image": images_to_return}
        return JsonResponse(response_data)
    else:
        return JsonResponse({"success": False})


@csrf_exempt
def upload_photo_for_seminar(request):
    if request.method == "POST" and request.FILES["photo"]:
        photo = request.FILES["photo"]
        only_for_seminar.data_protection(photo)
        description = only_for_seminar.get_description()
        response_data = {"description": description}
        return JsonResponse(response_data)
    else:
        return JsonResponse({"success": False})


@csrf_exempt
def update(request):
    if request.method == "POST":
        data = json.loads(request.body)
        # image = data.get("Image")
        lat = data.get("Latitude")
        lng = data.get("Longitude")
        des = data.get("Description")
        id = data.get("UserID")
        only_for_seminar.write_csv(lat, lng, des, id)

        return JsonResponse({"success": True})
    else:
        return JsonResponse({"success": False})
