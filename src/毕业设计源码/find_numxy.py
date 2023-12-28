# coiding=utf-8
import numpy as np
import cv2
from keras.models import load_model
IMAGE_SIZE = 28


def where_num(frame):
    rois = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray2 = cv2.dilate(gray, element)
    gray2 = cv2.erode(gray2, element)
    edges = cv2.absdiff(gray, gray2)
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    ret_1, ddst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(
        ddst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 12 and h > 24:
            rois.append((x, y, w, h))
    return rois


def resize_image(image, height=28, width=28):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    BLACK = [0, 0, 0]
    ret, thresh2 = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV)
    constant = cv2.copyMakeBorder(
        thresh2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    onstant = cv2.copyMakeBorder(
        constant, 30, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(onstant, (height, width))


model = load_model('./final-model.h5')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    rois = where_num(frame)
    print rois
    if len(rois) > 0:
        for r in rois:
            x, y, w, h = r
            image = frame[y: y + h, x: x + w]
            # cv2.imshow('t',image)
            image = resize_image(image)
            cv2.imshow('rsize', image)
            image = image.reshape((3, IMAGE_SIZE, IMAGE_SIZE, 1))
            image = image.astype('float32')
            image /= 255
            result = model.predict_classes(image)
           # print result
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(result[0]), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('My Camera', frame)
    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
