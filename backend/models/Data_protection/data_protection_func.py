import cv2

from ultralytics import YOLO


def blur(image, box, border):
    # get box coordinates in (top, left, bottom, right) format
    x1, y1, x2, y2 = box.xyxy[0]
    object = image[int(y1) : int(y2), int(x1) : int(x2)]
    object = cv2.GaussianBlur(object, (55, 55), border)
    # put the blurred object into the original frame
    image[int(y1) : int(y2), int(x1) : int(x2)] = object
    
    return image
    
def data_protection(image):
    model_face = YOLO("yolov8n-face.pt")
    model_plate = YOLO("license_plate_detector.pt")
    results_face = model_face.predict(source=image, show=False)
    result_plate = model_plate.predict(source=image, show=False)

    for r in results_face:
        boxes = r.boxes
        for box in boxes:
            image = blur(image, box, 10)
        
    for r in result_plate:
        boxes = r.boxes
        for box in boxes:
            image = blur(image, box, 40)

    return image


# if __name__ == '__main__':
#     main()
