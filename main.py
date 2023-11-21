Python 3.9.7 (tags/v3.9.7:1016ef3, Aug 30 2021, 20:19:38) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 


import cv2
import numpy as np

# Threshold to detect object (threshold  is at what point we should detect it as an actual object)
thres = 0.60
nms_threshold = 0.2

# img = cv2.imread('ali2.jpg')
# img = cv2.imread('street.jpg')

# video capture is to read the live video from your system camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
# cap.set(4, 720)
# cap.set(10, 70)

# insted of putting names one by one in list create a class called classNames
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)

# config and weight path
configPath = 'Models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'Models/frozen_inference_graph.pb'

# detection model is a trained model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    if success is True:
        # bbox is bounding box through the info we will create a box around object and specify it
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        # confs = list(np.array(confs).reshape(1, -1)[0])
        # confs = list(map(float, confs))
        print(classIds, bbox)
        print(len(bbox))

# create a rectangle around the object and put name and confidence value inside
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        c = cv2.waitKey(10)
        if c == 27:
            break
    else:
      break
cv2.destroyAllWindows()
cap.release()

