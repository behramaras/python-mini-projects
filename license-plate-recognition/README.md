# License Plate Recognition System

This project is a **Python-based License Plate Recognition System** that detects vehicle license plates from images, extracts plate numbers using OCR, and manages vehicle access through a SQLite database.

It simulates a **parking gate / access control system** by allowing or denying vehicle entry based on registration records and logging entry–exit times.

---

## Features

* License plate detection from vehicle images using **OpenCV**
* Plate number extraction using **Tesseract OCR**
* Vehicle registration and authorization with **SQLite**
* Automatic **entry & exit logging**
* Command-line menu–driven interface
* Batch processing of vehicle images from a folder

---

## Technologies Used

* **Python 3**
* **OpenCV**
* **NumPy**
* **pytesseract**
* **SQLite**
* **imutils**

---

## Project Structure

```
license_plate_recognition/
│
├── license_plate_recognition.py
├── license_plate.db
├── images/
│   ├── plate01.jpg
│   ├── plate02.jpg
│   └── plate03.jpg
└── README.md
```

> The `images/` folder contains sample vehicle images used for plate recognition.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/behramaras/python-mini-projects.git
cd python-mini-projects/license_plate_recognition
```

### Install required libraries

```bash
pip install opencv-python numpy pytesseract imutils
```

### Install Tesseract OCR

**macOS**

```bash
brew install tesseract
```

Make sure the path is correct in the code:

```python
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
```

---

## How to Run

```bash
python license_plate_recognition.py
```

You will see a menu:

```
1) Registration Management
2) Run Plate Recognition
3) Vehicle Logs
4) Exit
```

---

## How It Works

1. Vehicle images are read from the `images/` folder
2. Plates are detected using contour detection
3. Plate text is extracted using OCR
4. The system checks if the plate is registered
5. Entry or exit is logged automatically in the database

---

## Database Tables

### Registered_Vehicles

| Owner | Plate |
| ----- | ----- |

### Vehicle_Log

| Plate | Action | Entry_Time | Exit_Time |
| ----- | ------ | ---------- | --------- |

---

## Future Improvements

* Live camera (webcam) support
* GUI interface
* Improved OCR accuracy
* REST API integration
* Cloud-based database

---

## Author

**Behram Aras**

GitHub: [https://github.com/behramaras](https://github.com/behramaras)

---

## License

This project is for educational and academic purposes.
