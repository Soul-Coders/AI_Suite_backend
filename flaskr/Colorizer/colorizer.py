# research: https://arxiv.org/pdf/1603.08511.pdf


import numpy as np
import cv2
import os

base_path = os.path.abspath(os.path.dirname(__file__))


class Colorizer:
    # height and width of the image 480, 600 being the default resolution
    def __init__(self, height=480, width=600):
        '''
        Initialize the model
        '''
        (self.height, self.width) = (height, width)

        # Reading the models using dnn model of CV2 by providing the paths to prototext and caffemodel
        self.color_model = cv2.dnn.readNetFromCaffe(
            os.path.join(base_path, 'model/colorization_deploy_v2.prototxt'),
            caffeModel=os.path.join(base_path, "model/colorization_release_v2.caffemodel")
        )

        # pre-trained clusters centroids from the model which are provided as numpy dump
        cluster_centers = np.load(os.path.join(base_path, 'model/pts_in_hull.npy'))
        # take the transpose of the the centers and reshape
        cluster_centers = cluster_centers.transpose().reshape(2, 313, 1, 1)

        # setting the ab layer
        self.color_model.getLayer(self.color_model.getLayerId('class8_ab')).blobs = [cluster_centers.astype(np.float32)]

        # fill a value of 2.606 in an array of shape 1x313 in the `con8_313_rh` layer
        self.color_model.getLayer(self.color_model.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]
    
    def process_img(self, img):
        '''
        Performs image manipulation
        '''
        # store the image as a numpy array
        np_img = np.fromfile(img, np.uint8)

        # get the image
        self.img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # resize the image
        self.img = cv2.resize(self.img, (self.width, self.height))

        # colorize the image
        self.colorize_img()

        # save the colored image
        new_img_name = 'created_by_daily_ai_suite_' + img.filename
        cv2.imwrite(new_img_name, self.out_img)

        return new_img_name

    
    def colorize_img(self):
        '''
        Colorize the image:
        # L channel has the greyscale information; which serves as the input to the system
        # AB channels has the color information; which is the output to the system
        # L to AB is mapped using CNN
        # then the predicted AB is concatenated with the input image to get the output
        '''
        # normalize the image: swap blue and red channels because the model is trained in RGB and opencv reads as BGR
        normalized_img = (self.img[:, :, [2, 1, 0]] * 1/255).astype(np.float32)

        # Formally the work is to be done in CIELAB (LAB) color space;
        # RGB color space to CIELAB (LAB) color space (spearates the color space and the luminance)
        img_lab = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2Lab)
        l_channel = img_lab[:, :, 0]  # or channel 0 which contains the luminance information

        # Now changing the image resolution of the image as the model is trained on 224 X 224 image resolution
        resized_img_lab = cv2.cvtColor(cv2.resize(normalized_img, (224, 224)), cv2.COLOR_RGB2Lab)
        resized_l_channel = resized_img_lab[:, :, 0] - 50

        # set the input to resized l channel
        self.color_model.setInput(cv2.dnn.blobFromImage(resized_l_channel))
        result = self.color_model.forward()[0, :, :, :].transpose((1, 2, 0))

        # result is in 256 X 256 resolution so resizing it
        resized_result = cv2.resize(result, (self.width, self.height))

        self.out_img = np.concatenate((l_channel[:, :, np.newaxis], resized_result), axis=2)
        self.out_img = np.clip(cv2.cvtColor(self.out_img, cv2.COLOR_LAB2BGR), 0, 1)  # clipping all values between 0 and 1
        self.out_img = np.array(self.out_img * 255, dtype=np.uint8)
