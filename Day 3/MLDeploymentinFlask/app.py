from flask import Flask, render_template, request
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from model import load
    import re, base64, cv2
    import numpy as np
    from crop import borders

app = Flask(__name__)
model = load.init()


@app.route('/')
def index():
    return render_template("index.html")


def model_predict(img_path, model):
    img = cv2.imread(img_path, 0)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (28, 28))
    img = cv2.GaussianBlur(img, (3, 3), -1)
    top, bottom, b = borders(img, 0)
    img = img[top:bottom]
    cv2.imwrite('abc.png', img)
    img = img.reshape(-1, 28, 28, 1) / 255
    preds = model.predict(img)
    return preds


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    imgData = str((imgData), 'utf-8')
    imgstr = re.search(r'base64,(.*)', imgData).group(1)

    with open('output.png', 'wb') as output:
        output.write(np.fromstring(base64.b64decode(imgstr), np.uint8))
    file_path = 'output.png'
    result = model_predict(file_path, model)
    response = str(np.argmax(result))
    return response


if __name__ == "__main__":
    app.run(debug=True)
