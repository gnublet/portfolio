import cv2
import numpy as np

# Load the face, eye, nose, cascade file
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_mouth.xml')

# Check if the face cascade file has been loaded
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

#Check if eye cascade file has been loaded correctly
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')

#Check if nose file loaded correctly
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')

#Check smile
if mouth_cascade.empty():
    raise IOError('Unable to load the smile cascade classifier xml file')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the scaling factor
scaling_factor = 0.5

# Loop until you hit the Esc key
while True:
    # Capture the current frame and resize it
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale image
    face_rect = face_cascade.detectMultiScale(gray, 1.3, 5)
    #1.3 scale multiplier for each stage
    #5 min number of neighbors that each candidate rectangle should have so that we retain it.

    #run eye, nose, mouth detector within each face rectangle
    for (x,y,w,h) in face_rect:#(lowerleft (x,y), width, height)
        #grab current region of interest in both color and grayscale
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #run eye detector in grayscale roi
        eye_rects = eye_cascade.detectMultiScale(roi_gray, 1.3, 3)

        #nose detector
        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.3,5)

        #smile detector 
        mouth_rects = mouth_cascade.detectMultiScale(roi_gray, 2,10)

        #draw blue circles around eyes
        for (x_eye, y_eye, w_eye, h_eye) in eye_rects:
            center = (int(x_eye + .5*w_eye), int(y_eye + .5*h_eye))
            radius = (int(.3*(w_eye + h_eye)))
            color = (0,0,255)
            thickness = 1
            cv2.circle(roi_color, center, radius, color, thickness)

        #draw red rectangle around nose
        for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
            cv2.rectangle(roi_color, (x_nose, y_nose), (x_nose+w_nose, y_nose+h_nose), (255,0,0), 1)

        #draw rect around mouth
        for (x_mouth, y_mouth, w_mouth, h_mouth) in mouth_rects:
            cv2.rectangle(roi_color, (x_mouth, y_mouth), (x_mouth+w_mouth, y_mouth+h_mouth), (255,255,0), 1)
    
    # Draw rectangles on the image
    for (x,y,w,h) in face_rect:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Display the image
    cv2.imshow('Face Detector', frame)

    # Check if Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
