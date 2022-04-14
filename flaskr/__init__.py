from flask import (
    Flask, jsonify, request
)
from PIL import Image

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

    @app.route("/ocr", methods=["POST"])
    def ocr():
        image = request.files['image']
        txt = get_text(image)
        return jsonify({'text': txt})
        
    @app.route("/enhance_compress", methods=["POST"])
    def enhance_or_compress():
        value = request.form.get('value')
        file = request.files.get('image')
        img = Image.open(file)
        img.save(file.filename, quality=int(value))
        return jsonify({"new_img": file.filename})
    
    @app.route("/colors", methods=["POST"])
    def colors():
        image = request.files['image'].read()
        scheme = generate_color(image)
        return jsonify({'colors': scheme})

    @app.route("/colorize", methods=["POST"])
    def colorize():
        image = request.files['image']
        colorizer = Colorizer()
        colored = colorizer.process_img(image)
        Image.open(colored).show()
        return f"{image.filename}"

    @app.route("/summarize", methods=["POST"])
    def summarize():
        link = request.form.get('link')
        summary = generate_summary(link)
        return jsonify({'summary': summary})

    @app.route("/bgremoval", methods=["POST"])
    def rem():
        image = request.files.get('image')
        img = remove_bg(image)
        img.show()
        return f"{image.filename}"

    return app


app = create_app()
app.run(debug=True)
