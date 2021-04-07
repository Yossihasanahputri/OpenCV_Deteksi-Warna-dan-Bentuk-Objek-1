import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cap.set(3, 200)
cap.set(4, 250)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 255 // 2, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(frame, 100, 200)

    lower_Green = np.array([50, 100, 100])
    upper_Green = np.array([70, 255, 255])

    lower_Blue = np.array([100, 100, 100])
    upper_Blue = np.array([120, 255, 255])

    lower_Red = np.array([0, 100, 100])
    upper_Red = np.array([3, 255, 255])

    mask1 = cv2.inRange(hsv, lower_Green, upper_Green)
    mask2 = cv2.inRange(hsv, lower_Blue, upper_Blue)
    mask3 = cv2.inRange(hsv, lower_Red, upper_Red)

    cnts1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    cnts3 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)

    for c in cnts1:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) >= 6:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 4)
            cv2.putText(frame, "Daun Singkong", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    for c in cnts3:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) >= 6:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 4)
            cv2.putText(frame, "Bukan Daun Singkong", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    for c in cnts3:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) >= 6:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 4)
            cv2.putText(frame, "Bukan Daun Singkong", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Get", frame)
    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    image = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    # show the images
    cv2.imshow("hsv", hsv)
    cv2.imshow("gray", gray)
    cv2.imshow("Get2", image)
    cv2.imshow("binary", binary)
    cv2.imshow("edges", edges)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()