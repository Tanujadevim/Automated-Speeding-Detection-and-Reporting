import cv2
import dlib
import time
import math
import os
import easyocr

# Cascade file paths
carCascade = cv2.CascadeClassifier("vech.xml")
noPlateCascadePath =r"indian_license_plate.xml"
noPlateCascade = cv2.CascadeClassifier(noPlateCascadePath)
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
NUMBER_PLATE_FOLDER = os.path.join(OUTPUT_FOLDER, "number_plate_output")  # New folder for number plate images
FACE_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "face_output")  # New folder for face images
HELMET_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "helmet_output")  # New folder for helmet images
MASK_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "mask_output")  # New folder for mask images
SPEEDING_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "speeding_output")  # New folder for speeding vehicle images
RECORDED_VIDEO_FOLDER = os.path.join(OUTPUT_FOLDER, "recorded_video")  # New folder for recorded video

# Initialize the easyocr reader with the 'en' language
reader = easyocr.Reader(['en'])

# Maximum Speed Limit
MAX_SPEED_LIMIT = 30

# Estimate speed function
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

# (Rest of the functions remain the same)

if __name__ == '__main__':
    # Create output folders if they don't exist
    os.makedirs(NUMBER_PLATE_FOLDER, exist_ok=True)
    os.makedirs(FACE_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(HELMET_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(SPEEDING_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(RECORDED_VIDEO_FOLDER, exist_ok=True)

    # Video writer setup moved outside the loop
    out = cv2.VideoWriter(os.path.join(RECORDED_VIDEO_FOLDER, 'outTraffic.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        # (Rest of the code remains the same)

        cv2.imshow('result', image)

        out.write(image)

        if cv2.waitKey(1) == 27:
            break

    # Save data outside the loop
    cv2.destroyAllWindows()
    out.release()
