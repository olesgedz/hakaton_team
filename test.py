import csv
import argparse
import logging
import time
import streamlink
import matplotlib.pyplot as plt
from tf_pose import common

import cv2
import numpy as np
import os
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common

def csv_write():
    with open('data.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar=';', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
        spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


parser = argparse.ArgumentParser(description='tf-pose-estimation run')
parser.add_argument('--image', type=str, default='./images/p1.jpg')
parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

args = parser.parse_args()

path = './imgs/normal'

files = []
files_n = 0
for r, d, f in os.walk(path):
    for file in f:
        if not '.DS' in file:
            files.append(os.path.join(r, file))
            files_n+=1

for f in files:
    print(f)


w, h = model_wh(args.resize)
if w == 0 or h == 0:
     e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

human_r = []
for image in files:
    image = common.read_imgfile(image, None, None)
    dim = (432, 368)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    number = 0
    for human in humans:
        print(number/files_n     * 100)
        print("\n")
        number+=1
        pair  = [0, 0]
        max = human.get_face_box(dim[0], dim[1])
        if max != None:
            for part in human.body_parts.values():
                pair[0] += part.x
                pair[1] += part.y - max.get("y", "")
        human_r.append(pair)
        print(human_r)

for human in human_r:
    print(human[0], human[1])