import dlib
import cv2

to_bw = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def convert_cv2rect_to_dlibrect(x, y, w, h):
    return dlib.rectangle(x, y, x + w, y + h)

def convert_cv2rect_to_dlibrects(arrays):
    return [convert_cv2rect_to_dlibrect(x, y, w, h) for x, y, w, h in arrays]

def get_detections(image, detector, **kwargs):
    return detector.detectMultiScale(image, **kwargs)

def get_crop_from_rect(image, rect):
    x, y ,w, h = rect
    return image[y:(y+h),x:(x+w)].copy()

def get_crops_from_rectangles(image, rectangles):
    return [get_crop_from_rect(image, rec) for rec in rectangles]

def mark_detections(image, detections):
    for x, y, w, h in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0xff, 0), 4)

def detect_and_mark(image, detector):
    detections = get_detections(image, detector)
    mark_detections(image, detections)
    return detections

def get_shape_points(image, shape_predictor, rectangles):
    res = []
    for rect in rectangles:
        res.append(shape_predictor(image, rect))
    return res

def mark_points(image, points):
    for point in points:
        cv2.circle(image, (point.x, point.y), 2, (0,0, 0xff), 2)

# def get_shape_points(image, shape_predictor, rect):
#     return shape_predictor(image, rect)

def main():
    cv2.namedWindow("Screen", cv2.WINDOW_GUI_NORMAL)
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    image = cv2.imread("davi.jpeg")

    detections = detect_and_mark(image, face_detector)

    for detection in detections:
        detection = convert_cv2rect_to_dlibrect(*detection)
        sp = shape_predictor(image, detection)
        for point in sp.parts():
            cv2.circle(image, (point.x, point.y), 2, (0,0, 0xff), 2)
    
    cv2.imshow("Screen", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
