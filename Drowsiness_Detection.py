from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def process(frame,detect,predict,thresh,frame_check,flag,lStart,lEnd,rStart,rEnd):
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame,"--ALERT--",(20,50),
							cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
				#print ("Drowsy")
		else:
			flag = 0
	#cv2.waitKey(1)
	return frame

def drowsy():
	thresh = 100
	frame_check = 0
	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor(".\shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
	flag = 0
	cap = cv2.VideoCapture('drows.mp4')
	while (cap.isOpened()):
		ret, frame = cap.read()
		frame = process(frame,detect,predict,thresh,frame_check,flag,lStart,lEnd,rStart,rEnd)
		ret, buffer = cv2.imencode('.jpg', frame)
		frame = buffer.tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		if (cv2.waitKey(1) & 0xFF == ord('q')):
			break

	cap.release()
	cv2.destroyAllWindows()
