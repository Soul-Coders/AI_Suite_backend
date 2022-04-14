from PIL import Image
import cv2
from numpy import asarray
import inference

def remove_bg(image):
    image = Image.open(image)
    # print(type(image))
    im_pil = image
    image = asarray(image)    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img = inference.predict(image)
    # im_PIL.putalpha(img)
    # Image.open(im_PIL).show()

    return "MEh"


    #return send_file('results/detected.png', mimetype='image/png')