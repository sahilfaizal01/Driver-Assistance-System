import cv2
ss = cv2.CascadeClassifier('Stop Sign Classifier/stop_sign.xml')

def stops(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    SS = ss.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in SS:
       img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    return img



def stop():
    cap = cv2.VideoCapture('Stop Sign Classifier/stop.mp4')
    cap.set(4, 480)
    cap.set(3, 360)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = stops(frame)
        ret, buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()