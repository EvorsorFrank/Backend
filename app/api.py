from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, Path, HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

@app.get('/')
def hello_world():
    return 'Hello World'


@app.get('/hello')
def hello_cow():
    return 'Hello betch'

