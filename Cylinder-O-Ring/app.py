from flask import Flask, jsonify, request
from model import WM
from datetime import datetime
from PIL import Image
import io
import time
import base64
import json
import cv2
import numpy as np
import os

app = Flask(__name__)

model = WM()
model.load_network()

class Myclass:
    b64_string = ""
    cords = 0
    boolll = False
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))
@app.route("/model/predict", methods=["GET", "POST"])
def predict():

    data = json.loads(request.data)
    img = data.get("")
    #imageFile = data.get("imageFileName")
    imgFromcs = stringToImage(img)
    imgFromcs2 = base64.b64decode(img)
    jpg_as_np = np.frombuffer(imgFromcs2, dtype=np.uint8)
    image1 = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    numpyImage = np.array(imgFromcs)
    print("Image converted to numpy successfuly oring server")
    #jpg_img = cv2.imencode('.jpg', image1)
    #cv2.imshow("jpg_img",jpg_img)
    #cv2.waitkey(0)
    start=time.time()    
    cords, returnImage = model.get_predictions(numpyImage)
    obj = Myclass()
    obj.b64_string
    jpg_img = cv2.imencode('.jpg', returnImage)    
    obj.b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    obj.num = 5
    obj.boolll = False    
    return jsonify(b64_string = obj.b64_string, num=obj.num, boolll=cords)
##    image=base64.b64decode(str(img))
##    decoded_img = cv2.imdecode(np.frombuffer(image, np.uint8),-1)
##
##    result=model.run_ocr(decoded_img[:,:,1:4],vis=False)
##
##
##    return jsonify(result)
##
@app.route("/server", methods=["GET", "POST"])
def server():
    return "running"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
