from flask import Flask, render_template, Response
import cv2,glob
from process import *
from stop_class import *
from videosdetection import *
from Drowsiness_Detection import *
from gesture_control import *
app = Flask(__name__,static_folder="static",template_folder="templates")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ped')
def pedestrian_watch():
    return render_template('pedestrian_watch.html')

@app.route('/tired')
def tired():
    return render_template('drowsy.html')

@app.route('/gest')
def gest():
    return render_template('gesture.html')

@app.route('/lanes')
def lanes():
    return render_template('lane.html')

@app.route('/stops')
def stops():
    return render_template('stop.html')

@app.route('/ped/video_feed')#pedestrian
def video_feed():
	return Response(generator(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/lanes/lane_video_feed')
def lane_video_feed():
	return Response(lane(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stops/stop_video_feed')
def stop_video_feed():
	return Response(stop(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gest/gest_video_feed')
def gest_video_feed():
	return Response(gester(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tired/drowsy_video_feed')
def drowsy_video_feed():
    return Response(drowsy(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,threaded=True)

