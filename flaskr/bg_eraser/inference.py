import os
import cv2
import numpy as np
from PIL import Image
import onnxruntime

base_path = os.path.abspath(os.path.dirname(__file__))

model_path = os.path.join(base_path, "model/modnet.onnx")

def predict(im):

    ref_size = 512

    def get_scale_factor(im_h, im_w, ref_size):
        if max(im_h,im_w) < ref_size or min(im_h,im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w/im_h*ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h/im_w*ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw%32
        im_rh = im_rh - im_rh%32

        x_scale_factor = im_rw/im_w
        y_scale_factor = im_rh/im_h

        return x_scale_factor, y_scale_factor


    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    im = (im - 127.5) / 127.5
    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size=ref_size)

    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)

    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype('float32')

    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    matte = (np.squeeze(result[0])*255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)

    matte = Image.fromarray(matte)

    return matte