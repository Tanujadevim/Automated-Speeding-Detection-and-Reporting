# Automated Speeding Detection and Reporting 🚗💨
### *RoadSense AI: Transforming Safety with Computer Vision in Automated Speeding Detection and Traffic Reporting*

A real-time traffic monitoring system that uses computer vision to detect speeding vehicles, recognize number plates, detect faces, helmets, and masks — all via webcam.

---

## 📌 Project Overview

This system integrates multiple computer vision techniques into a single unified pipeline:

| Feature | Description |
|--------|-------------|
| 🚗 Vehicle Tracking | Tracks multiple vehicles using Dlib correlation tracker |
| ⚡ Speed Estimation | Estimates vehicle speed based on pixel displacement |
| 🪪 Number Plate Recognition | Detects & reads Indian number plates using EasyOCR |
| 👤 Face Detection | Detects driver faces using Dlib frontal face detector |
| ⛑️ Helmet Detection | Detects helmets using custom Haar Cascade |
| 😷 Mask Detection | Detects masks using Haar Cascade |
| 🎥 Video Recording | Saves annotated output video automatically |

---

## 🗂️ Project Structure

```
Automated-Speeding-Detection-and-Reporting/
│
├── src/
│   ├── main_system.py              # ✅ Main complete system (use this to run)
│   ├── face_only.py                # Face + helmet + mask detection only
│   └── anni_detect.py              # Alternate version of main system
│
├── cascades/
│   ├── vech.xml                    # Vehicle detection cascade
│   ├── indian_license_plate.xml    # Indian number plate cascade
│   ├── india-noplate.xml           # Additional plate cascade
│   ├── haarcascade_frontalface_default.xml  # Face detection cascade
│   ├── haarcascade_helmet.xml      # Helmet detection cascade
│   ├── haarcascade_mcs_nose.xml    # Mask/nose detection cascade
│   └── nose_data.xml               # Nose data cascade
│
├── output/                         # Auto-created when you run the system
│   ├── number_plate_output/        # Detected number plate images
│   ├── face_output/                # Detected face images
│   ├── helmet_output/              # Detected helmet images
│   ├── mask_output/                # Detected mask images
│   ├── speeding_output/            # Speeding vehicle images
│   └── recorded_video/             # Output video (outTraffic.avi)
│
├── docs/
│   └── Automation_Speed.pdf        # Full project presentation (40 slides)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/Automated-Speeding-Detection-and-Reporting.git
cd Automated-Speeding-Detection-and-Reporting
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python src/main_system.py
```

- Allow webcam access when prompted
- The system will start detecting in real-time
- Press **`ESC`** to stop
- All outputs are automatically saved to the `output/` folder

---

## 🧠 How It Works

1. **Webcam** captures live video frames
2. **Haar Cascades** detect vehicles and number plates in each frame
3. **Dlib correlation tracker** tracks each vehicle across frames
4. **Speed estimation** calculates km/h based on pixel displacement between frames
5. **EasyOCR** reads the text from detected number plate regions
6. **Dlib face detector** identifies driver faces
7. **Haar Cascades** check for helmets and masks on detected faces
8. All detections are **saved** to organized output folders with timestamps
9. Full video is **recorded** and saved to `output/recorded_video/`

---

## 📁 Output Folders (Auto-created)

| Folder | Contents |
|--------|---------|
| `number_plate_output/` | Cropped number plate images with timestamp |
| `face_output/` | Cropped face images with timestamp |
| `helmet_output/` | Cropped helmet detections |
| `mask_output/` | Cropped mask detections |
| `speeding_output/` | Images of speeding vehicles with speed label |
| `recorded_video/` | Full annotated video (outTraffic.avi) |

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| Python 3.x | Core language |
| OpenCV (cv2) | Video capture, image processing, cascade detection |
| Dlib | Face detection, vehicle correlation tracking |
| EasyOCR | Number plate text recognition |
| Math | Speed calculation |
| OS | Folder management & file paths |
| Platform | PyCharm |

---

## 📊 Project Presentation

The full project presentation (40 slides) is available in `docs/Automation_Speed.pdf` covering Abstract, System Analysis, Implementation, Training Process, and Outputs.

---

## 👩‍💻 Team Members

| Name |
|------|
Anjali
Poojita 
Tanuja Devi. M
Siri

---

## 📄 License

This project is for educational purposes.
