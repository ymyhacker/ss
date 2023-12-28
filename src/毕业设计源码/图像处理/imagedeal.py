# coding=utf-8
import cv2
import numpy as np


def where_num(frame):
    rois = []
    # 灰度处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 二次腐蚀处理
    gray2 = cv2.dilate(gray, element)
    gray3 = cv2.dilate(gray2, element)
    # cv2.imshow("dilate", gray3)
    # 二次膨胀处理
    gray2 = cv2.erode(gray2, element)
    gray2 = cv2.erode(gray2, element)
    # cv2.imshow("erode", gray2)
    # 膨胀腐蚀做差
    edges = cv2.absdiff(gray, gray2)
    # cv2.imshow("absdiff", edges)
    # 使用算子进行降噪
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # cv2.imshow("sobel", dst)
    # 选择阀值对图片进行二值化处理
    ret_1, ddst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    # 寻找图片中出现过得轮廓
    im, contours, hierarchy = cv2.findContours(
        ddst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在保存的轮廓里利用宽度和高度进行筛选
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
    ret, thresh2 = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
    constant = cv2.copyMakeBorder(
        thresh2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    onstant = cv2.copyMakeBorder(
        constant, 30, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(onstant, (height, width))


image = cv2.imread('./test.jpg')
num = where_num(image)
x, y, w, h = num[0]
num7 = image[y:y + h, x:x + w]

num1 = resize_image(num7)
print num1.shape
cv2.imshow('num1', num1)


num3 = cv2.imread('./test7.jpeg')
print num3.shape
cv2.imshow('num3', num3)
# for r in num:
#     x, y, w, h = r
#     numimage = image[y:y + h, x:x + w]
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv2.imshow('thenum', numimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
