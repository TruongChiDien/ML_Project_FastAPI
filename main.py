from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Response
import shutil
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os
import glob
from fastapi.templating import Jinja2Templates
import torch
import moviepy.editor as moviepy


imageExtensions = ['jpg', 'png', 'jpeg']
videoExtensions = ['mp4', 'avi', 'webm']


HEAD_HTML = """
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="background-color: cornsilk;">
<center>
"""

BEGIN_TABLE = """
<table align="center">
'<tr>
    <th><h4 style="font-family:Arial">Ảnh gốc</h4></th>
    <th><h4 style="font-family:Arial">Ảnh dự đoán</h4></th>
</tr>'
"""

END_TABLE = """
</table>
"""

app = FastAPI()
app.mount("/files_uploaded", StaticFiles(directory="files_uploaded"), name="files_uploaded")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/result", StaticFiles(directory="result"), name="result")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='static/without_oxford.pt')



@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_files(files: List[UploadFile] = File(...)):
    fileNames = [file.filename for file in files]
    fileExtensions = [filename.split('.')[1] for filename in fileNames]
    
    imagePaths, imagePredPaths, images = [], [], []
    videoPaths, videoPredPaths, videos = [], [], []

    for i, file in enumerate(files):
        if fileExtensions[i] in imageExtensions:
            image = await file.read()
            image = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image_rgb)
            imagePath = 'files_uploaded/' + fileNames[i]
            pillow_image = Image.fromarray(image_rgb)
            pillow_image.save(imagePath)
            imagePaths.append(imagePath)
            imagePredPaths.append('result/' + fileNames[i].split('.')[0] + '.jpg')

        elif fileExtensions[i] in videoExtensions:
            videoPath = 'files_uploaded/' + fileNames[i]
            with open(videoPath, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
            videoPaths.append(videoPath)
            videoPredPaths.append('result/' + fileNames[i])

            video_frames = []
            videoCap = cv2.VideoCapture(videoPath)
            ret, frame = videoCap.read()
            while ret:
                video_frames.append(frame)
                ret, frame = videoCap.read()
            fps = videoCap.get(cv2.CAP_PROP_FPS)
            width = videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            videoCap.release()
            videos.append((video_frames, fps, width, height))
            # del video_frames

        else:
            return HTTPException(status_code=402)

    # detect command line

    for i in range(0, len(images), 50):
        img_preds = model(images[i:min(i+50, len(images))])
        for img_pred, img_path in zip(img_preds.render(), imagePredPaths):
            pillow_image = Image.fromarray(img_pred)
            pillow_image.save(img_path)

    videoConvertPaths = []
    for (frames, fps, width, height), videoPredPath in zip(videos, videoPredPaths):
        print(len(frames), int(width), int(height))
        videoWritter = cv2.VideoWriter(videoPredPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
        for i in range(0, len(frames), 50):
            frame_preds = model(frames[i: min(i+50, len(frames))])
            for frame in frame_preds.render():
                videoWritter.write(frame)
        videoWritter.release()
        real_path = Path(videoPredPath)
        convertPath = str(real_path).replace(real_path.stem, real_path.stem + "_convert")
        clip = moviepy.VideoFileClip(videoPredPath)
        clip.write_videofile(convertPath)
        videoConvertPaths.append(convertPath)
        
    imagesTable = getImagesTable(imagePaths, imagePredPaths)
    videosTable = getVideosTable(videoPaths, videoConvertPaths)


    content = HEAD_HTML + \
        """
        <marquee width="525" behavior="alternate">
            <h1 style="color:red;font-family:Arial">Đây là kết quả!</h1>
        </marquee>
        <br>
        <br>
        """ + \
        BEGIN_TABLE + imagesTable + videosTable + END_TABLE +\
        """
        <br>
        <form method="get" action="/">
        <button type="submit">Home</button>
        </form>
        </body>
        """

    return content






@app.post("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def main():
    # Delete files have been in upload folder
    for up, pred in zip(glob.glob('files_uploaded/*'), glob.glob('result/*')):
        os.remove(up)
        os.remove(pred)

    content = HEAD_HTML + \
    """
    <marquee width="525" behavior="alternate">
        <h1 style="color:red;font-family:Arial">Xin mời gửi ảnh lên!</h1>
    </marquee>
    <br>
    <br>
    """

    ori_paths = ['static/original.jpg']
    pred_paths = ['static/predict.jpg']
    content = content + BEGIN_TABLE + getImagesTable(ori_paths, pred_paths) + END_TABLE

    content = content + """
    <br/>
    <br/>
    <form  action="/uploadfile/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    </body>
    """
    return content


def getImagesTable(ori_paths, pred_paths):  
    table = ""

    for ori, pred in zip(ori_paths, pred_paths):
        table = table + \
        """
        <tr>
            <td><img width="480" src="/{}"></td>
            <td><img width="480" src="/{}"></td>
        </tr>
        """.format(ori, pred)
    return table



def getVideosTable(ori_paths, pred_paths):
    table = ""

    for ori, pred in zip(ori_paths, pred_paths):
        extension = ori.split('.')[1]
        table = table + \
        """
        <td><video controls width="480" src="/{}" type=video/{}></video></td>
        <td><video controls width="480" src="/{}" type=video/{}></video></td>
        """.format(ori, extension, pred, extension)
    return table
    
