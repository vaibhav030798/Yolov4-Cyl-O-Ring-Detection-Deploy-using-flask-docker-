import numpy as np
import cv2
import os
import glob
import time
import sys
import matplotlib.pyplot as plt

confidenceThreshold = 0.5
NMSThreshold = 0.5

modelConfiguration = 'data/yolov4-custom.cfg'
modelWeights = 'data/yolov4-custom_best.weights'

labelsPath = 'data/obj.names'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

class WM():
    def load_network(self):
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net = net
    def predict(self, img):
        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        labels = open(labelsPath).read().strip().split('\n')
        a = 0
        image = img
        (H, W) = image.shape[:2]
        #Determine output layer names
        layerName = net.getUnconnectedOutLayersNames() 
        #layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
        net.setInput(blob)
        layersOutputs = net.forward(layerName)
        boxes = []
        confidences = []
        classIDs = []
        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        #print('classIDs is ', classIDs)
        #Apply Non Maxima Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
        #print(len(detectionNMS))
        cords = []
        if len(idxs)>0:   
           # loop over the indexes we are keepin
            for i in idxs.flatten():
                  detection_dict={}
                  (x,y)=(boxes[i][0], boxes[i][1])
                  (w,h)=(boxes[i][2], boxes[i][3])                  
                  text = "{}".format(labels[classIDs[i]])
                  confidence=round(confidences[i],2)
                  ptx,pty=x+(w/2),y+(w/2)
                  a,b=x,y
                  c,d=x+w, y+h
                  H,W=image.shape[0], image.shape[1]
                  detection_dict["okng"]=text
                  detection_dict["x1"]=str(round(ptx/W,2))
                  detection_dict["y1"]=str(round(pty/H,2))
                  detection_dict["width"]=str(round(w/W,2))
                  detection_dict["height"]=str(round(h/H,2))
                  cords.append(detection_dict)
                  #print('cords', cords)
        return {"_ResponseList": cords}

    def get_predictions(imgPath, save = False):
        img = cv2.imread(imgPath)      
        label_map={"ok":0,"ng":1}
        final_preds = []
        #image_name = os.path.basename(imgPath)
        #img = cv2.cvtColor(np.array(imgPath), cv2.COLOR_RGB2BGR)        
        #print(cords["_ResponseList"])
        
##        f = open(r"C:\Users\Admin\Desktop\New Text Document.txt", "a")
##        f.write("Now the file has more content!")
##        f.close()
##        path = "D:\starion_bolt\save_img"
##        cv2.imwrite(os.path.join(path, "python.bmp"),img)
##        cv2.waitKey(0)
        cords = self.predict(img)
        #print(cords)
        H, W, _ = img.shape        
        for ind,c in enumerate(cords["_ResponseList"]):
            #print(c)
            label = c["okng"]
            #print(label)
            final_preds.append(label)
            #################### To show the results ##############################
            x1 = float(c["x1"]) * W
            y1 = float(c["y1"]) * H
            w = float(c["width"]) * W
            h = float(c["height"]) * H
            xmin, ymin, xmax, ymax = (
                int(x1) - int(w / 2),
                int(y1) - int(h / 2),
                int(x1) + int(w / 2),
                int(y1) + int(h / 2),
            )
            xmin, ymin, xmax, ymax = int(x1)-int(w/2), int(y1)-int(h/2), int(x1)+int(w/2), int(y1)+int(h/2)
##            print('xmin',xmin)
##            print('ymin',ymin)
##            print('xmax',xmax)
##            print('ymax',ymax)               
            if label == "ng":
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 6)            
            elif label == "ok":
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 6)
        img = cv2.resize(img , (512,512))
        cv2.imshow("img", img)
        cv2.waitKey(0)
##        path = r"C:\Users\MSI 1\OneDrive\Desktop\save_img"
##        cv2.imwrite(os.path.join(path, "python.bmp"),img)
        #cv2.waitKey(0)  
        if "ng" in final_preds:           
            return False, img
        elif "ok" in final_preds and len(final_preds)>0:          
            return True, img        
        elif "ok" in final_preds:        
            return True, img
        else:          
            return False, img

if __name__ == "__main__":
    start = time.time()
    model = WM()    
    model.load_network()
    for imgPath in glob.glob("test\*.bmp"):
        #print(imgPath)
        model.get_predictions(imgPath)
    #cv2.imshow("orignal", imagePath)
    #cv2.waitKey()    
    end = time.time()
    print("--- %s seconds ---" % (time.time() - start))
    
    



