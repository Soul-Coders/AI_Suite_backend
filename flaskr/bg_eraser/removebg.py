from PIL import Image
import cv2
from numpy import asarray
from . import inference

def remove_bg(image):
    image = Image.open(image)
    im_pil = image
    image = asarray(image)    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img = inference.predict(image)
    im_pil.putalpha(img)
    return im_pil