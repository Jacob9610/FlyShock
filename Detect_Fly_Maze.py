#pip install FreeSimpleGUI run this command in the command prompt before running the python script

import numpy as np
import cv2
import FreeSimpleGUI as sg
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time

############### CHOOSE VIDEO FILE INTERACTIVELY ###################
Tk().withdraw()  # Hide the root Tkinter window
vid = askopenfilename(title="Select a video file", filetypes=[("Video Files", "*.mp4 *.mov *.avi")])
if not vid:
    print("No file selected. Exiting...")
    exit()

detectFileName = vid + '.csv'  # file that saves object data

########## PROGRAM VARIABLES ################################################
medianFrames = 25
skipFrames = 300
BLUR = 5
THRESH = 56
DELAY = 1
THICK = 3
X_REZ = 640
Y_REZ = 480
MIN_AREA = 50
MAX_SINGLE_AREA = 80
MAX_MULTI_AREA = 300
DISPLAY_REZ = (640, 480)
PROCESS_REZ = (320, 240)

print('Process Resolution', PROCESS_REZ)

############# DETECT OUTPUT ##################
detectHeader = 'FRAME,ID,XC,YC,AREA,MULTI_FLAG'
MAX_COL = 6
FRAME, ID, XC, YC, AREA, MULTI_FLAG = range(MAX_COL)
detectArray = np.empty((0, MAX_COL), dtype='int')

# Crossing detection variables
crossing_line = PROCESS_REZ[0] // 2
crossing_count = 0
fly_positions = {}

# Create Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# ----------- FUNCTION: Calculate Median Frame -----------
# @param vid: path to the video file
# @param medianFrames: number of frames to sample for median calculation
# @param PROCESS_REZ: resolution to process frames
# @return: median grayscale frame used as static background reference

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

# ----------- FUNCTION: Apply Processing to Frame -----------
# @param frame: input BGR frame
# @param mask: binary mask defining region of interest (ROI)
# @param medianFrame: background reference frame for subtraction
# @return: masked grayscale image and processed binary mask of detected foreground

def process_frame(frame, mask, medianFrame):
    frame_resized = cv2.resize(frame, PROCESS_REZ)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    fgmask = fgbg.apply(masked)
    blur = cv2.GaussianBlur(fgmask, (BLUR, BLUR), 0)
    roi_mean = np.mean(masked)
    dynamic_thresh = int(roi_mean * 0.8)
    _, binary = cv2.threshold(blur, dynamic_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return masked, binary

# ----------- FUNCTION: Create Binary Mask for ROI -----------
# @param process_res: resolution of processing frame
# @param roi_coords: tuple of (x, y, width, height) for rectangular ROI
# @return: binary mask image

def create_mask(process_res, roi_coords):
    mask = np.zeros((process_res[1], process_res[0]), dtype=np.uint8)
    x, y, w, h = roi_coords
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

# ----------- FUNCTION: Initialize Video Capture -----------
# @param path: file path of the video
# @param start_frame: number of frames to skip before processing
# @return: cv2.VideoCapture object, or None if failure

def initialize_video_capture(path, start_frame):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return cap

# ----------- FUNCTION: Save Detection Results to CSV -----------
# @param filename: name of the CSV file
# @param data: numpy array of detection data
# @param header: string of CSV header
# @return: None

def save_results(filename, data, header):
    np.savetxt(filename, data, header=header, delimiter=',', fmt='%d')

# ----------- FUNCTION: Main Program Loop -----------
# @return: None. Displays and processes video until completion or exit.
# Saves processed frame data to CSV file at the end of processing.

def run_detection():
    global detectArray, frameCount

    medianFrame = getMedian(vid, medianFrames, PROCESS_REZ)
    if medianFrame is None:
        return

    mask = create_mask(PROCESS_REZ, (40, 20, 300, 400))
    cap = initialize_video_capture(vid, skipFrames)
    if cap is None:
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameCount = skipFrames
    start_time = time.time()

    while cap.isOpened():
        loop_start = time.time()
        key = chr(cv2.waitKey(DELAY) & 0xFF)
        if key == 'q':
            break


        ret, frame = cap.read()
        if not ret:
            break

        frameCount += 1

        # Skip every 2nd frame (50% frame reduction). Change 2 to 3 for more speedup.
        if frameCount % 2 != 0:  # Skip odd frames
           continue


        maskedGray, binaryIM = process_frame(frame, mask, medianFrame)

        # Timing estimate
        elapsed = time.time() - start_time
        frames_done = frameCount - skipFrames
        fps = frames_done / elapsed if elapsed > 0 else 0
        est_remaining = (total_frames - frameCount) / fps if fps > 0 else 0
        print(f"Frame: {frameCount}/{total_frames}, FPS: {fps:.2f}, ETA: {int(est_remaining // 60)} min {int(est_remaining % 60)} sec", end='\r')


        cv2.imshow('Masked Frame', cv2.resize(maskedGray, DISPLAY_REZ))
        cv2.imshow('Foreground Mask', cv2.resize(binaryIM, DISPLAY_REZ))

    cap.release()
    cv2.destroyAllWindows()

    if frameCount > skipFrames:
        print('\nDone with video. Saving detection file...')
        save_results(detectFileName, detectArray, detectHeader)
    else:
        print('\nCould not open or read frames from video:', vid)

# ----------- START PROGRAM -----------
print("\n\nUse '+' and '-' keys to change object detect threshold by 1")
print("Hold shift while pressing '+' or '-' to change threshold by 10\n")
run_detection()

