import os
import cv2

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.plotting import Annotator
from glob import glob

#Load model
model_face = YOLO("yolov8n-face.pt")
model_plate = YOLO('license_plate_detector.pt')
folder_path = './before_blurring/'
show_img = False
images = glob(os.path.join(folder_path + "*"))
for img in images:
    filename = os.path.basename(img)
    results_face = model_face.predict(source = img, show = False)
    result_plate = model_plate.predict(source = img, show = False)
    img = cv2.imread(img)

    for r in results_face:
        # annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            human = img[int(y1): int(y2), int(x1): int(x2)]
            human = cv2.GaussianBlur(human, (55, 55), 10)
            # put the blurred human into the original frame
            img[int(y1): int(y2), int(x1): int(x2)] = human
        # img = annotator.result()  

    for r in result_plate:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            plate = img[int(y1): int(y2), int(x1): int(x2)]
            plate = cv2.GaussianBlur(plate, (55, 55), 40)
            # put the blurred human into the original frame
            img[int(y1): int(y2), int(x1): int(x2)] = plate
    
        cv2.imwrite(f"./result/{filename}", img)
        if show_img == True:
            cv2.imshow('Face Blurring', img)  
            if cv2.waitKey(0):
                break