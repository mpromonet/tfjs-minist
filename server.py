from flask import Flask, json, request, redirect, url_for, send_from_directory
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

api = Flask(__name__,static_url_path='',static_folder='.')
model=tf.keras.models.load_model('mnist')

def runmodel(img):
    img = cv2.resize(img,(28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    predict = model.predict(img)

    value = tf.math.argmax(predict,1)[0].numpy()
    proba = predict[0][value]

    return {"value": value.item(), "proba": proba.item()}

@api.route('/')
def root():
    return api.send_static_file('index.html')

@api.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = secure_filename(file.filename)

        file.save(filename)
        img = cv2.imread(filename)

        return json.dumps(runmodel(img))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@api.route('/predict', methods=['POST'])
def predict():
    content = request.stream.read()
    img = cv2.imdecode(np.fromstring(content, dtype=np.uint8),cv2.IMREAD_COLOR)
    print(img)

    return json.dumps(runmodel(img))    


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    api.run(host='0.0.0.0', port=port)
