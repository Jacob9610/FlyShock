#pip install FreeSimpleGUI run this command in the command prompt before running the python script

import numpy as np
import cv2
import FreeSimpleGUI as sg

############### PUT YOUR VIDEO LINK HERE ###################
vid = './9-26_RG_20R_54T.mp4'
detectFileName = 'test.csv'  # file that saves object data

########## PROGRAM VARIABLES ################################################
medianFrames = 25  # number of random frames to calculate median frame brightness
skipFrames = 300  # give video image autobrightness (AGC) time to settle
BLUR = 3  # blur differenced images to remove holes in objects
THRESH = 56  # apply threshold to blurred object to create binary detected objects
DELAY = 1
THICK = 3  # bounding rectangle thickness

X_REZ = 640
Y_REZ = 480  # viewing resolution
MIN_AREA = 10  # min area of object detected
MAX_SINGLE_AREA = 80  # max area of one object detected
MAX_MULTI_AREA = 300  # max area of multiple objects detected
DISPLAY_REZ = (640, 480)  # display resolution
PROCESS_REZ = (320, 240)  # processing resolution, reduce size to speed up processing tghis may be replaced with threads so that it can process faster 

print('Process Resolution', PROCESS_REZ)

############# DETECT OUTPUT ##################
detectHeader = 'FRAME,ID,XC,YC,AREA,MULTI_FLAG'
MAX_COL = 6
FRAME, ID, XC, YC, AREA, MULTI_FLAG = range(MAX_COL)
detectArray = np.empty((0, MAX_COL), dtype='int')  # cast as int since most features are int and it simplifies usage

# Crossing detection variables
crossing_line = (PROCESS_REZ[0] // 2)-40 # Define the horizontal line at the center of the frame
crossing_count = 0  # Counter for crossings
fly_positions = {}  # Dictionary to track last known position of each fly

def getMedian(vid, medianFrames, PROCESS_REZ):
    # Open Video
    print('openVideo:', vid)
    cap = cv2.VideoCapture(vid)
    maxFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('maxFrame', maxFrame)

    # Check if the video opened correctly
    if not cap.isOpened():
        print(f"Error: Cannot open video {vid}")
        return None

    # Randomly select N frames
    print('calculating median')
    frameIds = skipFrames + (maxFrame - skipFrames) * np.random.uniform(size=medianFrames)
    frames = []  # Store selected frames in an array
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame at position {fid}. Check video codec or file.")
            continue  # Skip to the next frame

        # Resize and convert to grayscale
        try:
            colorIM = cv2.resize(frame, PROCESS_REZ)
        except cv2.error as e:
            print(f"Resize failed for frame {fid} with error: {e}")
            continue  # Skip to the next frame

        grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)
        frames.append(grayIM)

    if len(frames) == 0:
        print("Error: No frames were successfully read.")
        return None

    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)  # Calculate the median along the time axis
    cap.release()
    return medianFrame


######### MAIN PROGRAM ######################################################################################################################################
print("\n\nUse '+' and '-' keys to change object detect threshold by 1")
print("Hold shift while pressing '+' or '-' to change threshold by 10\n")

# Create median frame
medianFrame = getMedian(vid, medianFrames, PROCESS_REZ)

# Create a mask with the same dimensions as the grayscale image
mask = np.zeros((PROCESS_REZ[1], PROCESS_REZ[0]), dtype=np.uint8)

# Define a region of interest (ROI) if applicable, e.g., a rectangle:
roi_x, roi_y, roi_width, roi_height = 40, 20, 300, 400  # Example coordinates
cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), 255, -1)  # Fill rectangle with white

cap = cv2.VideoCapture(vid)
cap.set(cv2.CAP_PROP_POS_FRAMES, skipFrames)  # start movie past skipFrames
frameCount = skipFrames

while cap.isOpened():
    # read key, test for 'q' quit
    key = chr(cv2.waitKey(DELAY) & 0xFF)  # pause x msec

    if key == 'q':
        break
    elif key == '=':
        THRESH += 1
        print('Thresh:', THRESH)
    elif key == '+':
        THRESH += 10
        print('Thresh:', THRESH)
    elif key == '-' and THRESH > 1:
        THRESH -= 1
        print('Thresh:', THRESH)
    elif key == '_' and THRESH > 11:
        THRESH -= 10
        print('Thresh:', THRESH)

    # get image
    ret, colorIM = cap.read()
    if not ret:  # check to make sure there was a frame to read
        break
    frameCount += 1

    # capture frame, subtract median brightness frame, apply binary threshold
    colorIM = cv2.resize(colorIM, PROCESS_REZ)
    grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)  # convert color to grayscale image

    # Apply the mask to the grayscale image
    maskedGrayIM = cv2.bitwise_and(grayIM, grayIM, mask=mask)

    diffIM = cv2.absdiff(maskedGrayIM, medianFrame)  # Calculate absolute difference of current frame and the median frame
    blurIM = cv2.blur(diffIM, (BLUR, BLUR))
    ret, binaryIM = cv2.threshold(blurIM, THRESH, 255, cv2.THRESH_BINARY)  # threshold image to make pixels 0 or 255 or black and white

    # get contours
    contourList, hierarchy = cv2.findContours(binaryIM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # all contour points

    # draw bounding boxes around objects
    objCount = 0  # used as object ID in detectArray
    multiFlag = 0  # set if multi objects detected
    for objContour in contourList:  # process all objects in the contourList
        area = int(cv2.contourArea(objContour))  # find obj area
        if area > MIN_AREA and area < MAX_MULTI_AREA:  # only detect objects with acceptable area
            PO = cv2.boundingRect(objContour)
            x0 = PO[0]
            y0 = PO[1]
            x1 = x0 + PO[2]
            y1 = y0 + PO[3]
            xc = int((x1 - x0) / 2 + x0)  # x-center of object
            yc = int((y1 - y0) / 2 + y0)  # y-center of object



            if area < MAX_SINGLE_AREA:
                cv2.rectangle(colorIM, (x0, y0), (x1, y1), (0, 255, 0), THICK)  # GREEN rectangle
            else:
                cv2.rectangle(colorIM, (x0, y0), (x1, y1), (0, 0, 255), THICK)  # RED rectangle
                multiFlag = 1

            # Track crossings
            if objCount not in fly_positions:
                fly_positions[objCount] = yc  # Initialize the fly's last position

            # Check if fly has crossed the line
            if fly_positions[objCount] < crossing_line <= yc:  # Crossing from left to right
                crossing_count += 1
                print(f"Fly {objCount} crossed bottom up.")
            elif fly_positions[objCount] > crossing_line >= yc:  # Crossing from right to left
                crossing_count -= 1
                print(f"Fly {objCount} crossed from top down.")

            fly_positions[objCount] = xc  # Update the fly's last position

            # Save detection parameters
            parm = np.array([[frameCount, objCount, xc, y0, area, multiFlag]], dtype='int')  # create parameter vector
            detectArray = np.append(detectArray, parm, axis=0)  # add parameter vector to detectArray
            objCount += 1  # increment object count

    # Display crossing count on the video
    cv2.putText(colorIM, f"Crossing Count: {crossing_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.line(colorIM, (0, crossing_line), (PROCESS_REZ[0], crossing_line), (255, 0, 0), 2)  # Horizontal line


    # Shows results
    cv2.imshow('Masked Frame', cv2.resize(maskedGrayIM, DISPLAY_REZ))
    cv2.imshow('colorIM', cv2.resize(colorIM, DISPLAY_REZ))  # display image
    cv2.imshow('binaryIM', cv2.resize(binaryIM, DISPLAY_REZ))  # display binary image

if frameCount > 0:
    print('Done with video. Saving new csv file and exiting program')
    np.savetxt(detectFileName, detectArray, header=detectHeader, delimiter=',', fmt='%d')
    cap.release()
else:
    print('Could not open video', vid)
cv2.destroyAllWindows()
