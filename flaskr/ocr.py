import pytesseract
from PIL import Image
from numpy import asarray
import cv2

def get_text(image):
    image = Image.open(image)
    image = asarray(image)
    image = cv2.resize(image, None, fx=2, fy=2)
    config = '--oem 3 --psm 11'
    txt = pytesseract.image_to_string(image, config=config, lang='eng')
    cleaned_txt = txt.replace('\n\n', ' ').replace('\n', '')

    return cleaned_txt