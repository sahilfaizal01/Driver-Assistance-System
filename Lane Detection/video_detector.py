import matplotlib.pyplot as plt
import cv2
import numpy as np

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])

def process(image):
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0,height),
        (width/2, height/2),
        (width,height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_img, 100, 120)
    masked_img = region_of_interest(canny_image,
                                    np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(masked_img, rho=2, theta=np.pi / 180, threshold=100, lines=np.array([]), minLineLength=40,
                            maxLineGap=25)
    image_with_lines = draw_lines(image, lines)
    return image_with_lines

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img,lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0],img.shape[1], 3),dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=10)

    merged_img = cv2.addWeighted(img, 0.8,blank_image,1,0.0)
    return merged_img

def generator():
    cap = cv2.VideoCapture('Lane Detection/test2.mp4')
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


