from flask import Flask, send_file
from captcha.image import ImageCaptcha
import random
import string
import io
from PIL import ImageFilter

app = Flask(__name__)

def random_text(length=5):
    """Generate a random string of letters and digits."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

@app.route('/captcha')
def captcha():
    text = random_text(5)
    print(text)
    
    # Specify custom fonts and font sizes to vary the appearance.
    # Make sure these font paths are valid on your system.
    fonts = ['arial.ttf', 'times.ttf']  
    font_sizes = [50, 52, 54]
    
    # Create an ImageCaptcha instance with additional parameters.
    image_captcha = ImageCaptcha(width=200, height=64, fonts=fonts, font_sizes=font_sizes)
    image = image_captcha.generate_image(text)

    image.save("captcha.png", "PNG")
    
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
