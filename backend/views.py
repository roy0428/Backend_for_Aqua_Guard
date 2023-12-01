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


def original(photo):
    image = Image.open(photo)
    np_photo = np.array(image)
    np_photo = np.rot90(np_photo, k=3)
    np_photo = cv2.cvtColor(np_photo, cv2.COLOR_RGB2BGR)
    return np_photo


def mask(ortho_image):
    return functions.inference.main(ortho_image)


@csrf_exempt
def upload_photo(request):
    if request.method == "POST" and request.FILES["photo"]:
        start_time = time.time()
        photo = request.FILES["photo"]
        original_image = original(photo)
        masked_image = mask(original_image)
        _, buffer = cv2.imencode(".jpeg", masked_image)
        end_time = time.time()
        print("Total Runtime:", end_time - start_time, "s")
        return HttpResponse(buffer.tobytes(), content_type="image/jpeg")
    else:
        return JsonResponse({"success": False})


@csrf_exempt
def upload_video(request):
    if request.method == "POST" and request.FILES["video"]:
        start_time = time.time()
        video = request.FILES["video"]
        video_bytes = video.read()
        temp_path = os.path.join("backend/results", "temp_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        video_capture = cv2.VideoCapture(temp_path)
        image = test_combine.main(video_capture)
        _, buffer = cv2.imencode(".jpeg", image)
        end_time = time.time()
        print("Total Runtime:", end_time - start_time, "s")
        return HttpResponse(buffer.tobytes(), content_type="image/jpeg")
    else:
        return JsonResponse({"success": False})


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
        # TODO
        # preprocessed_photo = blur(photo)
        # description = gpt(preprocessed_photo)
        description = "returned description"
        response_data = {"description": description}
        return JsonResponse(response_data)
    else:
        return JsonResponse({"success": False})
    
@csrf_exempt
def update(request):
    if request.method == "POST":
        data = json.loads(request.body)
        image = data.get("Image")
        lat = data.get("Latitude")
        lng = data.get("Longitude")
        des = data.get("Description")
        id = data.get("UserID")
        only_for_seminar.write_csv(image, lat, lng, des, id)
        
        return JsonResponse({"success": True})
    else:
        return JsonResponse({"success": False})
