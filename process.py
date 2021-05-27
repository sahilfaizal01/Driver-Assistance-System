from flask import Flask, render_template, Response
import cv2,glob

classNames = []
classFile = 'Pedestrian Detection/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

configPath = 'Pedestrian Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'Pedestrian Detection/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

def process(img):

    classIds, confs, bbox = net.detect(img,confThreshold=0.5)

    for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,0,255),thickness=3)
        cv2.putText(img,classNames[classId - 1].upper(),(box[0] + 10,box[1] + 30),
                cv2.FONT_HERSHEY_DUPLEX,1,(255, 0, 0),2)

    return img

def generator():
    cap = cv2.VideoCapture('Pedestrian Detection/test.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = process(frame)
        ret, buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
