import torch
import cv2
import numpy as np
import shutil


model = torch.hub.load('ultralytics/yolov5', 'custom', path='static/pretrained.pt')

source_path = 'static\pedestrian.mp4'
video_path = 'files_uploaded/pedestrian.mp4'
videoPredPath = 'result/pedestrian.mp4'

    
video_frames = []
videoCap = cv2.VideoCapture(video_path)
ret, frame = videoCap.read()
while ret:
    video_frames.append(frame)
    ret, frame = videoCap.read()

fps = videoCap.get(cv2.CAP_PROP_FPS)
width = videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)
videoCap.release()

videoWritter = cv2.VideoWriter(videoPredPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
for i in range(0, len(video_frames), 50):
    frame_preds = model(video_frames[i: min(i+50, len(video_frames))])
    for frame in frame_preds.render():
        videoWritter.write(frame)

videoWritter.release()
