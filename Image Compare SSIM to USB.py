from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import relay_ft245r
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
# Create a VideoCapture object 
cap = cv2.VideoCapture(0)
# Capture a frame ret, img = cap.read() 
# Release the capture cap.release()
import time
nframes = 1024
interval = 6
for i in range(nframes):
    # capture
    ret, img = cap.read()
    # save file
    cv2.imwrite('captured_image.jpg', img)
    rb = relay_ft245r.FT245R()
    dev_list = rb.list_dev()
    # list of FT245R devices are returned
    if len(dev_list) == 0:
        print('No FT245R devices found')
        sys.exit()
        # Show their serial numbers
    for dev in dev_list:
        print(dev.serial_number)
    # Pick the first one for simplicity
    dev = dev_list[0]
    print('Using device with serial number ' + str(dev.serial_number))
    # load and crop the two input images
    imageA = cv2.imread("captured_image.jpg")
    y=100
    x=50
    h=130
    w=70
    cropA = imageA[y:y+h, x:x+w]
    imageB = cv2.imread("webcam_original.jpg")
    y=100
    x=50
    h=130
    w=70
    cropB = imageB[y:y+h, x:x+w]
    cv2.imshow("cropA", cropA)
    cv2.waitKey(2000)
    cv2.imshow("cropB", cropB)
    cv2.waitKey(2000)
    cv2.destroyWindow("cropB")
    cv2.destroyWindow("cropA")
    # convert the images to grayscale
    grayA = cv2.cvtColor(cropA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(cropB, cv2.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))
    # Compare SSIM and switch relay
    if score < 0.9:
            rb.connect(dev)
            rb.switchon(2)    
            time.sleep(1.0)
            rb.switchoff(2)
            time.sleep(1.0)
            dev.reset()
    # compute difference
    difference = cv2.subtract(cropA, cropB)
    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    # add the red mask to the images to make the differences obvious
    cropA[mask != 255] = [0, 0, 255]
    cropB[mask != 255] = [0, 0, 255]
    time.sleep(2.0)
    # show images
    cv2.imshow("The Difference", difference)
    #Timer to display detection image
    cv2.waitKey(3000)
    cv2.destroyWindow("The Difference") 
    #cv2.imshow("Thresh", thresh)
    # wait 5 seconds
    time.sleep(interval)
