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


def find_near_location(current_latitude, current_longitude):
    dataframe = pd.read_csv("/Users/rlab/Desktop/seminar/Development_Dataset.csv")
    images_to_return = {"ImageID": [], "Latitude": [], "Longitude": [], "UserID": []}
    for index, row in dataframe.iterrows():
        data_latitude = row["Latitude"]
        data_longitude = row["Longitude"]
        distance = math.dist(
            (current_latitude, current_longitude), (data_latitude, data_longitude)
        )
        if distance < 0.1:
            images_to_return["ImageID"].append(row["ImageID"])
            images_to_return["Latitude"].append(data_latitude)
            images_to_return["Longitude"].append(data_longitude)
            images_to_return["UserID"].append(row["UserID"])
    return images_to_return


def find_image(imageid):
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


def write_csv(image, lat, lng, des, user_id):
    data_path = "/Users/rlab/Desktop/seminar/Development_Dataset_Images/"
    csv_path = "/Users/rlab/Desktop/seminar/Development_Dataset.csv"
    dataframe = pd.read_csv(csv_path)
    
    bytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(bytes)).convert('RGB').transpose(Image.ROTATE_270)
    image_id = str(uuid.uuid4())
    output_path = os.path.join(data_path, f"{image_id}.jpeg")
    image.save(output_path)
    image.close()
    
    new_data = {
        "ImageID": image_id,
        "Date": date.today().strftime("%Y/%m/%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Latitude": lat,
        "Longitude": lng,
        "Description": des,
        "UserID": user_id,
    }
    # dataframe = dataframe.append(new_data, ignore_index=True)
    dataframe = pd.concat([dataframe, pd.DataFrame([new_data])], ignore_index=True)
    dataframe.to_csv(csv_path, index=False)


if __name__ == "__main__":
    find_near_location(53.8346519470215, -1.771159053)
