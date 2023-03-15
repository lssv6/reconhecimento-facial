from ddetector import *
import cv2, pickle
import numpy as np


face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# use with compute. 
face_descriptor = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

cam = cv2.VideoCapture(0)
cv2.namedWindow("Screen", cv2.WINDOW_GUI_NORMAL)
face_descs = []

f_descriptions = []
current_id = 0
limit = .5
recognized_subjects = set()
# trick
def best_fits(fd, fds):
    temp = []
    for sfd in fds:
        diferences = np.linalg.norm(fd - sfd, axis=1)
        minindex = np.argmin(diferences)
        temp.append(sfd[minindex])
    return np.array(temp)

with open("dados.bin","br") as data:
    f_descriptions = pickle.load(data)
while True:
    check, frame = cam.read()
    detections = get_detections(frame, face_detector)
    #no detections then we'll be going to do nothing
    if len(detections) == 0:
        continue
    detections = convert_cv2rect_to_dlibrects(detections)
    shape_points = get_shape_points(frame, shape_predictor, detections)
    chip  = dlib.get_face_chip(frame, shape_points[0])

    face_description = face_descriptor.compute_face_descriptor(chip)
    face_description = np.array(face_description)
    
    bf = best_fits(face_description, f_descriptions)

    diferences = np.linalg.norm(face_description - bf[1:], axis=1)

    minindex = np.argmin(diferences)
    minval = diferences[minindex]

    if minval < limit:
        print(f"recognized subject={minindex-1}")
        recognized_subjects.add(minindex-1)
    else:
        print("subject unrecognized :-(")
    
    cv2.imshow("Screen", chip)

    key = cv2.waitKey(1)
    if key == 27: #ESC
        break

print(recognized_subjects)
