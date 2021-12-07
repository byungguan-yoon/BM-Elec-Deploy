from fastapi import FastAPI, Request
from fastapi import templating
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel
import numpy as np
import json
import cv2

from starlette.responses import Response

from database import (
    fetch_last_one_result,
    fetch_one_result,
    fetch_all_results
)
import schemas
from rle2contours import rle2contours

templates = Jinja2Templates(directory="./static/templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name='static')

origins = ["http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})


@app.get("/monitoring") 
async def monitoring():
    response = await fetch_last_one_result()
    return response


@app.get("/sections/{section_id}")
async def patch(request: Request):
    return templates.TemplateResponse("patch.html",{"request":request})


@app.post("/onMask")
async def onMask(infor: schemas.Item):
    section_id = infor.section_id
    patch_id = infor.patch_id
    print(section_id)
    print(patch_id)
    im_png = await rle2contours(section_id, patch_id)
    res, im_png = cv2.imencode(".png", im_png)
    enc_img = np.fromstring(im_png, dtype = np.uint8)
    encoded_img = json.dumps(enc_img,cls=NumpyEncoder)
    return {'status': 'OK', 'images': encoded_img}


@app.get("/tables")
async def tables(request: Request):
    return templates.TemplateResponse("display_table.html",{"request":request})


@app.get("/stats")
async def stats():
    result = await fetch_all_results()
    response = []
    for i in range(len(result)):
        id = str(result[i]['_id'])
        date, time = result[i]['timestamp'].split(':')
        dict_tmp = {
            "id": id,
            "date": date,
            "time": time.replace('_', ':')
        }
        response.append(dict_tmp)
    return response


@app.get("/tables/{timestamp}")
async def tables(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})


@app.get("/searching/{timestamp}") 
async def searching(timestamp):
    document = {"timestamp": timestamp}
    response = await fetch_one_result(document)
    return response