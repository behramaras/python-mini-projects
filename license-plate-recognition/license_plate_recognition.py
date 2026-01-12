import sqlite3  # Used for database communication
import cv2      # Used for reading license plates from images
import imutils
import numpy as np
import pytesseract
import datetime
import os


# Path to Tesseract OCR (change if necessary)
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'


class LicensePlateSystem:
    def __init__(self, system_name):
        self.database = sqlite3.connect("license_plate.db")
        self.cursor = self.database.cursor()
        self.system_name = system_name
        self.running = True

    def recognize_plate(self):
        """Main function for recognizing license plates from images"""

        image_folder = "images"
        files = os.listdir(image_folder)
