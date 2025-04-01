#pip install FreeSimpleGUI run this command in the command prompt before running the python script

import numpy as np
import cv2
import FreeSimpleGUI as sg

############### PUT YOUR VIDEO LINK HERE ###################
vid = './9-26_RG_20R_54T.mp4'
detectFileName = vid + '.csv'  # file that saves object data

########## PROGRAM VARIABLES ################################################
medianFrames = 25  # number of random frames to calculate median frame brightness
skipFrames = 300  # give video image autobrightness (AGC) time to settle
BLUR = 5  # Increased blur to reduce noise
THRESH = 56  # apply threshold to blurred object to create binary detected objects
DELAY = 1
THICK = 3  # bounding rectangle thickness

X_REZ = 640
Y_REZ = 480  # viewing resolution
MIN_AREA = 50  # Increased min area to filter small noise
MAX_SINGLE_AREA = 80  # max area of one object detected
MAX_MULTI_AREA = 300  # max area of multiple objects detected
DISPLAY_REZ = (640, 480)  # display resolution
PROCESS_REZ = (320, 240)  # processing resolution

print('Process Resolution', PROCESS_REZ)

############# DETECT OUTPUT ##################
detectHeader = 'FRAME,ID,XC,YC,AREA,MULTI_FLAG'
MAX_COL = 6
FRAME, ID, XC, YC, AREA, MULTI_FLAG = range(MAX_COL)
detectArray = np.empty((0, MAX_COL), dtype='int')

# Crossing detection variables
crossing_line = PROCESS_REZ[0] // 2  # Define the horizontal line at the center of the frame
crossing_count = 0  # Counter for crossings
fly_positions = {}  # Dictionary to track last known position of each fly

# Create Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def getMedian(vid, medianFrames, PROCESS_REZ):
    print('openVideo:', vid)
    cap = cv2.VideoCapture(vid)
    maxFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('maxFrame', maxFrame)

    if not cap.isOpened():
        print(f"Error: Cannot open video {vid}")
        return None

    print('calculating median')
    frameIds = skipFrames + (maxFrame - skipFrames) * np.random.uniform(size=medianFrames)
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            colorIM = cv2.resize(frame, PROCESS_REZ)
        except cv2.error:
            continue
        grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)
        frames.append(grayIM)

    if len(frames) == 0:
        print("Error: No frames were successfully read.")
        return None

    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cap.release()
    return medianFrame

######### MAIN PROGRAM #########
print("\n\nUse '+' and '-' keys to change object detect threshold by 1")
print("Hold shift while pressing '+' or '-' to change threshold by 10\n")

# Create median frame
medianFrame = getMedian(vid, medianFrames, PROCESS_REZ)

# Create a mask
mask = np.zeros((PROCESS_REZ[1], PROCESS_REZ[0]), dtype=np.uint8)
roi_x, roi_y, roi_width, roi_height = 40, 20, 300, 400
cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), 255, -1)

cap = cv2.VideoCapture(vid)
cap.set(cv2.CAP_PROP_POS_FRAMES, skipFrames)
frameCount = skipFrames

while cap.isOpened():
    key = chr(cv2.waitKey(DELAY) & 0xFF)
    if key == 'q':
        break

    ret, colorIM = cap.read()
    if not ret:
        break
    frameCount += 1

    colorIM = cv2.resize(colorIM, PROCESS_REZ)
    grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)
    maskedGrayIM = cv2.bitwise_and(grayIM, grayIM, mask=mask)

    # Apply background subtraction
    fgmask = fgbg.apply(maskedGrayIM)
    
    # Apply Gaussian blur
    blurIM = cv2.GaussianBlur(fgmask, (BLUR, BLUR), 0)
    
    # Adaptive thresholding
    roi_mean = np.mean(maskedGrayIM)
    dynamic_thresh = int(roi_mean * 0.8)
    _, binaryIM = cv2.threshold(blurIM, dynamic_thresh, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binaryIM = cv2.morphologyEx(binaryIM, cv2.MORPH_OPEN, kernel)
    binaryIM = cv2.morphologyEx(binaryIM, cv2.MORPH_CLOSE, kernel)

    # Display the results
    cv2.imshow('Masked Frame', cv2.resize(maskedGrayIM, DISPLAY_REZ))
    cv2.imshow('Foreground Mask', cv2.resize(binaryIM, DISPLAY_REZ))

# Exit and save data after loop
cap.release()
cv2.destroyAllWindows()

if frameCount > skipFrames:
    print('Done with video. Saving detection file...')
    np.savetxt(detectFileName, detectArray, header=detectHeader, delimiter=',', fmt='%d')
else:
    print('Could not open or read frames from video:', vid)
