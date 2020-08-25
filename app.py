import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'


def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')
def otsu_threshold(Folha):
    gray_ori = cv2.cvtColor(Folha, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray_ori,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshed

def removeBackground(Folha, threshold_func):
    # apply threshold
    threshed = threshold_func(Folha)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel) # dilation
    return morphed

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    if len(files) != 0:
        files = os.listdir(app.config['UPLOAD_PATH'])
        print(files)
        files = files[-1]
        path = os.path.join("uploads", files)
        image2 = cv2.imread(path)
        Otsu_morphology= removeBackground(image2, otsu_threshold)
        image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB )
        L = image3[:,:,0].ravel()
        A = image3[:,:,1].ravel()
        B = image3[:,:,2].ravel()
        Otsu_morphology1=Otsu_morphology.ravel()
        print(A)
        X= image3.shape[0]
        Y= image3.shape[1]
        xs = X * list(range(0, Y))
        xs = [round(x/Y, 3) for x in xs]
        ys = np.array([Y * [i] for i in range(0,X)]).ravel()
        ys = [round(y/X, 3) for y in ys]
        
        df = pd.DataFrame({"L": L,"A": A, "B": B,"x": xs,"y": ys, "Otsu":Otsu_morphology1})
        df = df[df.Otsu != 255]
        kmeans = KMeans(n_clusters=2).fit(df[["A", "B", "x"]].values)
        df['predict'] = kmeans.predict(df[["A", "B", "x"]].values)
        num1 = len(df[df['predict'] == 1])
        num2 = len(df[df['predict'] == 0])
        num3 = round((num1/num2)*100, 3)
        if num3 > 100 :
            num3 = round((100/num3)*100, 3)

        df = df.iloc[1:5, :]
        jsonfiles = json.loads(df.to_json(orient='records'))
        titles = ['L', 'A', 'B', 'x', 'y', 'Otsu', 'predict']

        return render_template('index.html', files=files, tables=jsonfiles, titles=titles, num3=num3) # {% for file in files %} {% endfor %}
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


if __name__== '__main__':
    app.run(debug=True)