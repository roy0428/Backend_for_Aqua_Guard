import pandas as pd

# import ipdb
import math
from PIL import Image
import os
import base64
import cv2
import numpy as np
import time
import io
import uuid
from datetime import date
from datetime import datetime
from ultralytics import YOLO
from openai import OpenAI


def apply_blur(image, results):
    def blur(image, box):
        # get box coordinates in (top, left, bottom, right) format
        x1, y1, x2, y2 = box.xyxy[0]
        object = image[int(y1) : int(y2), int(x1) : int(x2)]
        object = cv2.GaussianBlur(object, (55, 55), 40)
        # put the blurred object into the original frame
        image[int(y1) : int(y2), int(x1) : int(x2)] = object
        return image

    for result in results:
        boxes = result.boxes
        for box in boxes:
            image = blur(image, box)

    return image


def data_protection(image):
    image = Image.open(image).transpose(Image.ROTATE_270)
    source_path = "/Users/rlab/Desktop/seminar/Development_Dataset_Images/cache.jpeg"
    image.save(source_path)
    model_face = YOLO("backend/models/Data_protection/yolov8n-face.pt")
    model_plate = YOLO("backend/models/Data_protection/license_plate_detector.pt")
    results_face = model_face.predict(source=source_path, show=False)
    result_plate = model_plate.predict(source=source_path, show=False)
    image = cv2.imread(source_path)

    image = apply_blur(image, results_face)
    image = apply_blur(image, result_plate)
    cv2.imwrite(source_path, image)
    return


def find_near_location(current_latitude, current_longitude):
    dataframe = pd.read_csv("/Users/rlab/Desktop/seminar/Development_Dataset.csv")
    images_to_return = {"ImageID": [], "Latitude": [], "Longitude": [], "UserID": []}
    for index, row in dataframe.iterrows():
        data_latitude = row["Latitude"]
        data_longitude = row["Longitude"]
        distance = math.dist(
            (current_latitude, current_longitude), (data_latitude, data_longitude)
        )
        if distance < 1:
            images_to_return["ImageID"].append(row["ImageID"])
            images_to_return["Latitude"].append(data_latitude)
            images_to_return["Longitude"].append(data_longitude)
            images_to_return["UserID"].append(row["UserID"])
    return images_to_return


def find_image(imageid):
    pd.set_option("display.max_colwidth", None)
    dataframe = pd.read_csv("/Users/rlab/Desktop/seminar/Development_Dataset.csv")
    row = dataframe[dataframe["ImageID"] == imageid]
    image = np.array(
        Image.open(
            os.path.join(
                "/Users/rlab/Desktop/seminar/Development_Dataset_Images",
                imageid + ".jpeg",
            )
        )
    )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpeg", image)
    images_to_return = {
        "Image": base64.b64encode(buffer.tobytes()).decode("utf-8"),
        "Date": row["Date"].to_string(index=False, header=False),
        "Time": row["Time"].to_string(index=False, header=False),
        "Description": row["Description"].to_string(index=False, header=False),
    }
    return images_to_return


def write_csv(lat, lng, des, user_id):
    data_path = "/Users/rlab/Desktop/seminar/Development_Dataset_Images/"
    csv_path = "/Users/rlab/Desktop/seminar/Development_Dataset.csv"
    cache_path = "/Users/rlab/Desktop/seminar/Development_Dataset_Images/cache.jpeg"

    image = Image.open(cache_path)
    image_id = str(uuid.uuid4())
    image.save(os.path.join(data_path, f"{image_id}.jpeg"))
    os.remove(cache_path)

    new_data = {
        "ImageID": image_id,
        "Date": date.today().strftime("%Y/%m/%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Latitude": lat,
        "Longitude": lng,
        "Description": des,
        "UserID": user_id,
    }
    dataframe = pd.read_csv(csv_path)
    dataframe = pd.concat([dataframe, pd.DataFrame([new_data])], ignore_index=True)
    dataframe.to_csv(csv_path, index=False)


def get_description():
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    api_key = "APIKEY"
    client = OpenAI(api_key=api_key)
    prompt = "No more than 100 words"
    system_content = """The GPT specializes in analyzing urban river photographs, focusing on environmental indicators. It identifies and describes elements like outfall, sewage litter, discoloration, obstructions, channel modifications, aeration, sensors, and wildlife in no more than 100 words. The GPT provides objective, factual observations from the photograph without speculative interpretations. The responses are concise, informative, and suitable for both environmental professionals and enthusiasts. The GPT will ask for clarifications if the photo's details are ambiguous, ensuring accurate and succinct analyses. The tone remains informative, catering to a varied audience interested in environmental observations of urban rivers."""
    image_path = "/Users/rlab/Desktop/seminar/Development_Dataset_Images/cache.jpeg"
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=200,
    )
    Description = response.choices[0].message.content
    return Description


if __name__ == "__main__":
    find_near_location(53.8346519470215, -1.771159053)
