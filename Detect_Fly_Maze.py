#pip install FreeSimpleGUI run this command in the command prompt before running the python script

import numpy as np
import cv2
import FreeSimpleGUI as sg
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import os
from datetime import datetime
import signal
import sys

############### CHOOSE VIDEO FILE INTERACTIVELY ###################
Tk().withdraw()  # Hide the root Tkinter window
vid = askopenfilename(title="Select a video file", filetypes=[("Video Files", "*.mp4 *.mov *.avi")])
if not vid:
    print("No file selected. Exiting...")
    exit()

video_start_time = datetime.fromtimestamp(os.path.getmtime(vid))
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
detectLog = []
crossing_line = PROCESS_REZ[1] // 2  # horizontal line
zone_counts = {"top": 0, "bottom": 0}
fly_positions = {}

# Frame crop range (set after crop selection)
start_crop_frame = 0
end_crop_frame = None

# Create Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# ----------- FUNCTION: Prompt for Experiment Info -----------
def prompt_experiment_metadata():
    layout = [
        [sg.Text('Top Zone Color:'), sg.Combo(['Green', 'Red', 'Blue'], key='top_color')],
        [sg.Text('Bottom Zone Color:'), sg.Combo(['Green', 'Red', 'Blue'], key='bottom_color')],
        [sg.Text('Number of Flies in Top Zone:'), sg.Input(key='top_flies')],
        [sg.Text('Number of Flies in Bottom Zone:'), sg.Input(key='bottom_flies')],
        [sg.Button('OK')]
    ]
    window = sg.Window('Experiment Setup', layout)
    values = {}
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            exit()
        if event == 'OK':
            try:
                zone_counts['top'] = int(values['top_flies'])
                zone_counts['bottom'] = int(values['bottom_flies'])
                if values['top_color'] and values['bottom_color']:
                    break
            except:
                pass  # ignore invalid input
    window.close()
    return values['top_color'], values['bottom_color']

# ----------- FUNCTION: Prompt for Crop Range -----------
def select_crop_range(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Failed to open video for cropping preview")
        return 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    layout = [
        [sg.Text("Select start and end time (in seconds)")],
        [sg.Text("Start:"), sg.Slider(range=(0, total_frames//30), orientation='h', key='start', resolution=1)],
        [sg.Text("End:"), sg.Slider(range=(0, total_frames//30), orientation='h', key='end', resolution=1)],
        [sg.Button("OK")]
    ]
    window = sg.Window("Video Crop Range", layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            exit()
        if event == "OK":
            start_sec = int(values['start'])
            end_sec = int(values['end'])
            break
    window.close()
    cap.release()
    return int(start_sec * 30), int(end_sec * 30)

# ----------- FUNCTION: Calculate Median Frame -----------
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

# ----------- FUNCTION: Frame Processing -----------
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

# ----------- FUNCTION: Mask Creation -----------
def create_mask(process_res, roi_coords):
    mask = np.zeros((process_res[1], process_res[0]), dtype=np.uint8)
    x, y, w, h = roi_coords
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

# ----------- FUNCTION: Initialize Video Capture -----------
def initialize_video_capture(path, start_frame):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return cap

# ----------- FUNCTION: Save Detection Results -----------
def save_results(filename, log, top_color, bottom_color):
    with open(filename, 'w') as f:
        f.write(f"# Video: {vid}\n")
        f.write(f"# Start time: {video_start_time}\n")
        f.write(f"# Top color: {top_color}\n")
        f.write(f"# Bottom color: {bottom_color}\n")
        f.write(f"# Initial Top: {zone_counts['top']}, Bottom: {zone_counts['bottom']}\n")
        f.write("Frame,Timestamp(s),Direction,TopCount,BottomCount\n")
        for row in log:
            f.write(",".join(map(str, row)) + "\n")

# ----------- FUNCTION: Handle Graceful Exit -----------
def handle_exit(signal_received=None, frame=None):
    print('\nInterrupt received. Saving data before exit...')
    save_results(detectFileName, detectLog, top_color, bottom_color)
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# ----------- FUNCTION: Main Program -----------
def run_detection():
    global detectArray, top_color, bottom_color, start_crop_frame, end_crop_frame

    top_color, bottom_color = prompt_experiment_metadata()
    start_crop_frame, end_crop_frame = select_crop_range(vid)

    medianFrame = getMedian(vid, medianFrames, PROCESS_REZ)
    if medianFrame is None:
        return

    mask = create_mask(PROCESS_REZ, (40, 20, 300, 400))
    cap = initialize_video_capture(vid, start_crop_frame)
    if cap is None:
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameCount = start_crop_frame
    start_time = time.time()

    fly_last_y = {}
    fly_id = 0

    while cap.isOpened():
        key = chr(cv2.waitKey(DELAY) & 0xFF)
        if key == 'q':
            break

        if frameCount > end_crop_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frameCount += 1
        if frameCount % 2 != 0:
            continue

        maskedGray, binaryIM = process_frame(frame, mask, medianFrame)

        contours, _ = cv2.findContours(binaryIM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_MULTI_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            yc = y + h // 2
            xc = x + w // 2

            if fly_id not in fly_last_y:
                fly_last_y[fly_id] = yc
                fly_id += 1
                continue

            last_y = fly_last_y[fly_id]
            if last_y < crossing_line <= yc:
                zone_counts['top'] -= 1
                zone_counts['bottom'] += 1
                timestamp = frameCount / fps
                detectLog.append([frameCount, round(timestamp, 2), "Top→Bottom", zone_counts['top'], zone_counts['bottom']])
            elif last_y > crossing_line >= yc:
                zone_counts['top'] += 1
                zone_counts['bottom'] -= 1
                timestamp = frameCount / fps
                detectLog.append([frameCount, round(timestamp, 2), "Bottom→Top", zone_counts['top'], zone_counts['bottom']])

            fly_last_y[fly_id] = yc
            fly_id += 1

        elapsed = time.time() - start_time
        frames_done = frameCount - start_crop_frame
        processing_fps = frames_done / elapsed if elapsed > 0 else 0
        est_remaining = (end_crop_frame - frameCount) / processing_fps if processing_fps > 0 else 0
        print(f"Frame: {frameCount}/{end_crop_frame}, FPS: {processing_fps:.2f}, ETA: {int(est_remaining // 60)} min {int(est_remaining % 60)} sec", end='\r')

        cv2.imshow('Masked Frame', cv2.resize(maskedGray, DISPLAY_REZ))
        cv2.imshow('Binary Mask', cv2.resize(binaryIM, DISPLAY_REZ))

    cap.release()
    cv2.destroyAllWindows()

    if detectLog:
        print('\nSaving crossing data...')
        save_results(detectFileName, detectLog, top_color, bottom_color)
    else:
        print('\nNo crossings detected.')

# ----------- START PROGRAM -----------
print("\n\nUse '+' and '-' keys to change object detect threshold by 1")
print("Hold shift while pressing '+' or '-' to change threshold by 10\n")
run_detection()