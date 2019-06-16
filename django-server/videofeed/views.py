from django.shortcuts import render
from .forms import UploadVideoFileForm
from django.http.response import StreamingHttpResponse
import argparse
import logging
import time
import streamlink
import matplotlib.pyplot as plt
import cv2
import numpy as np
import threading
from django.views.decorators.gzip import gzip_page

from videofeed.tf_pose.estimator import TfPoseEstimator
from videofeed.tf_pose.networks  import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

class VideoCamera(object):
    def __init__(self):
        self.w, self.h = model_wh('432x368')
        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(self.w, self.h))
        else:
            self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(100, 100))
        self.w, self.h = model_wh('432x368')
        self.streams = streamlink.streams('https://www.twitch.tv/zpinkman_')#https://www.twitch.tv/blondynkitezgraja')
        self.url = self.streams['360p'].url
        self.cam = cv2.VideoCapture(self.url)
        self.i = 0
        self.help = False
        (self.grabbed, self.frame) = self.cam.read()
        self.humans = self.e.inference(self.frame, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=4.0)
        self.dim = (656, 368)
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.cam.release()

    def get_frame(self):
        image = cv2.resize(self.frame, self.dim, interpolation = cv2.INTER_AREA)
        if self.i > 50:
            self.i = 0
            self.humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=4.0)
            image = TfPoseEstimator.draw_humans(image, self.humans, imgcopy=False)
        else:
            image = TfPoseEstimator.draw_humans(image, self.humans, imgcopy=False)
        if self.help == True:
            image = cv2.putText(image, 'ALARM', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,34,34), lineType = cv2.LINE_AA)
        self.i += 1
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.cam.read()

video = VideoCamera() 

# Create your views here.
def video_upload(request):
    if request.method == 'POST':
        print("BIBA")
    else:
        form = UploadVideoFileForm()
    return render(request, 'uploadvideo.html', {'form': form})

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip_page
def livefe(request):
    try:
        return StreamingHttpResponse(gen(video), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass
