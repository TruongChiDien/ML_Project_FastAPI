from collections import defaultdict
from fastapi import FastAPI, UploadFile, File
# from fastapi.datastructures import UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
from utils import draw_bb, model_predict, resize, non_max_suppression_fast

model = tflite.Interpreter('static/model.tflite')
model.allocate_tensors()

head_html = """
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="background-color: cornsilk;">
<center>
"""


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_files(file: UploadFile = File(...)):
    name = file.filename
    image = await file.read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_resized = resize(image)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    bb = model_predict(model, np.array(image_rgb, dtype=np.float32))
    draw_bb(image_resized, bb, name)
    # pillow_image = Image.fromarray(image_rgb)
    # pillow_image.save('static/' + name)

    content = head_html + \
        """
        <marquee width="525" behavior="alternate"><h1 style="color:red;font-family:Arial">Kết quả nè tml!</h1></marquee>
        """ + \
        get_html_image('static/' + name) + \
        """<br><form method="post" action="/">
        <button type="submit">Home</button>
        </form></body>"""

    return content



@app.post("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def main():
    content = head_html + """
    <marquee width="525" behavior="alternate"><h1 style="color:red;font-family:Arial">Gửi ảnh lên mày!</h1></marquee>
    """

    default_image = '978.jpg'
    default_path = 'static/images/' + default_image

    content = content + get_html_image(default_path)

    content = content + """
    <br/><br/>
    <form  action="/uploadfile/" enctype="multipart/form-data" method="post">
    <input name="file" type="file"><input type="submit"></form>
    </body>
    """
    return content



def get_html_image(image_path):  
    s = '<p style="text-align: center;";>"<img height="640" src="/' + \
        image_path + '" alt="default image" class="center"></p>'
    return s