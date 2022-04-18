from flask import (
    Flask, jsonify, request, send_file
)
from flask_cors import CORS

from PIL import Image
import os

TAG = 'created_by_daily_ai_suite_'

# 1. OCR (CV2/Neural)
# 2. Image Editing:
# 	    Remove Background
# 	    Image Enhance/Compress
# 	    Image Colorization
# 3. Youtube transcript summarizer
# 4. Extract colour scheme from image

from ocr import get_text
from colors import generate_color
from Colorizer.colorizer import Colorizer
from summarizer import generate_summary
from bg_eraser.removebg import remove_bg

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)

    @app.route("/ocr", methods=["POST"])
    def ocr():
        image = request.files.get('image')
        txt = get_text(image)
        return jsonify({'text': txt})
        
    @app.route("/enhance_compress", methods=["POST"])
    def enhance_or_compress():
        value = request.form.get('value')
        file = request.files.get('image')
        img = Image.open(file)
        img.save(TAG + file.filename, quality=int(value))
        try:
            return send_file(TAG + file.filename)
        finally:
            os.remove(TAG + file.filename)
    
    @app.route("/colors", methods=["POST"])
    def colors():
        image = request.files.get('image').read()
        scheme = generate_color(image)
        return jsonify({'colors': scheme})

    @app.route("/colorize", methods=["POST"])
    def colorize():
        image = request.files.get('image')
        colorizer = Colorizer()
        colored = colorizer.process_img(image)
        try:
            return send_file(colored)
        finally:
            os.remove(TAG + image.filename)

    @app.route("/summarize", methods=["POST"])
    def summarize():
        link = request.form.get('link')
        summary = generate_summary(link)
        return jsonify({'summary': summary})

    @app.route("/bgremoval", methods=["POST"])
    def rem():
        image = request.files.get('image')
        img = remove_bg(image)
        filename = TAG + image.filename[:-3] + 'png'
        img.save(filename)
        try:
            return send_file(filename)
        finally:
            os.remove(filename)
    return app


app = create_app()
app.run(debug=True)
