import cv2
import dlib
import time
import os
import easyocr  # Import the easyocr library

# Cascade file paths
faceCascadePath = r"haarcascade_frontalface_default.xml"
helmetCascadePath = r"haarcascade_helmet.xml"
maskCascadePath = r"haarcascade_mcs_nose.xml"

# Load the Dlib face detector
faceDetector = dlib.get_frontal_face_detector()

# Load the helmet and mask cascades
helmetCascade = cv2.CascadeClassifier(helmetCascadePath)
maskCascade = cv2.CascadeClassifier(maskCascadePath)

# Webcam video capture
video = cv2.VideoCapture(0)  # 0 corresponds to the default webcam, change it if you have multiple cameras

# Constant Declaration
WIDTH = 1000
HEIGHT = 1080
OUTPUT_FOLDER = r"output"
FACE_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "face_output")  # New folder for face images

# Create output folders if they don't exist
os.makedirs(FACE_OUTPUT_FOLDER, exist_ok=True)

# Initialize the easyocr reader with the 'en' language
reader = easyocr.Reader(['en'])


# Detect helmet on the face region
def detectHelmet(face_gray, face_roi):
    helmets = helmetCascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (hx, hy, hw, hh) in helmets:
        cv2.rectangle(face_roi, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 2)
        helmet_img = face_roi[hy:hy + hh, hx:hx + hw]
        img_name_helmet = f"helmet_{time.time()}.png"
        img_path_helmet = os.path.join(FACE_OUTPUT_FOLDER, img_name_helmet)
        cv2.imwrite(img_path_helmet, helmet_img)
    return face_roi


# Detect mask on the face region
def detectMask(face_gray, face_roi):
    masks = maskCascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (mx, my, mw, mh) in masks:
        cv2.rectangle(face_roi, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
        mask_img = face_roi[my:my + mh, mx:mx + mw]
        img_name_mask = f"mask_{time.time()}.png"
        img_path_mask = os.path.join(FACE_OUTPUT_FOLDER, img_name_mask)
        cv2.imwrite(img_path_mask, mask_img)
    return face_roi


# Face detection using Dlib
def detectFaces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceDetector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the face image to the specified output folder
        img_name_face = f"face_{time.time()}.png"
        img_path_face = os.path.join(FACE_OUTPUT_FOLDER, img_name_face)
        cv2.imwrite(img_path_face, image[y:y + h, x:x + w])

        # Crop and save the face region
        face_roi = image[y:y + h, x:x + w]
        cv2.putText(face_roi, str(time.ctime()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(img_path_face, face_roi)

        # Detect helmet and mask on the driver's face
        face_gray = gray[y:y + h, x:x + w]
        image[y:y + h, x:x + w] = detectHelmet(face_gray, image[y:y + h, x:x + w])
        image[y:y + h, x:x + w] = detectMask(face_gray, image[y:y + h, x:x + w])

    return image


if __name__ == '__main__':
    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        # Perform face detection
        resultImage = detectFaces(resultImage)

        cv2.imshow('result', resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
