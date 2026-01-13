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
        for file in files:
            print("Incoming vehicle image:", file)

            img = cv2.imread(os.path.join(image_folder, file))
            img = cv2.resize(img, (600, 400))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 13, 15, 15)

            edged = cv2.Canny(gray, 30, 200)
            contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            plate_contour = None

            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
                if len(approx) == 4:
                    plate_contour = approx
                    break

            if plate_contour is None:
                print("No license plate detected.")
                continue

            cv2.drawContours(img, [plate_contour], -1, (0, 0, 255), 3)

            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [plate_contour], 0, 255, -1)
            cropped = cv2.bitwise_and(gray, gray, mask=mask)

            coords = np.where(mask == 255)
            topx, topy = np.min(coords[0]), np.min(coords[1])
            bottomx, bottomy = np.max(coords[0]), np.max(coords[1])

            cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

            plate_text = pytesseract.image_to_string(cropped, config='--psm 11')
            plate_text = plate_text.strip('\n ,!')

            print("Detected Plate:", plate_text)

            cv2.imshow("Vehicle", img)
            cv2.imshow("Plate", cropped)

            self.process_plate(plate_text)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
