from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template, jsonify
import json
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import os
from skimage.transform import resize
import cv2
app = Flask(__name__, static_url_path='')

def model_predict(frame, model):
        img = resize(frame,(64,64,1))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img = img/255.0
        prediction = model.predict(img)
        prediction = model.predict_classes(img)
        return prediction


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/live')
def live():
# Train or test 
    mode = 'get'
    directory = 'C:/Users/nitis/OneDrive/Desktop/Remote Internship-2020/pythoncodes/data/'
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64, 64)) 
        cv2.imshow("Frame", frame)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("ROI", roi)
        k=os.listdir(directory+"/get")
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        if interrupt & 0xFF == ord('0'):
            x=len(k)
            cv2.imwrite(directory+'get/'+str(x)+'.jpg', roi)
            roi=cv2.imread(directory+'get/'+str(x)+'.jpg')
            model = load_model('myclassifier.h5')
            data=model_predict(roi,model)
            print(data)
            print("preds : "+str(data))
            ls=["zero","one","Two","Three","Four","Five"]
            result = ls[data[0]]
            print(result)
            return result

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']  
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        frame=cv2.imread(file_path)
        model = load_model('myclassifier.h5')
        print(frame)
        data=model_predict(frame,model)
        print(data)
        print("preds : "+str(data))
        ls=["zero","one","Two","Three","Four","Five"]
        result = ls[data[0]]
        print(result)
        return result
        return 
    return None

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)



