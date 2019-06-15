import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
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

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    dim = (w,h)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    head = 0
    print(w,h)
 
    for f in humans:
        pair = [0,0]
        max  = 0
        min  = 0
        for part in f.body_parts.values():
             print(part)
        for part in f.body_parts.values():
            if max < part.y:
                max = part.y
            if min > part.y:
                min = part.y
        i = 0
        for part in f.body_parts.values():
            #print(max)
            pair[0] += part.x
            pair[1] += (part.y - max) / (min - max)
            i+=1
        pair[1] /= i 
        print(pair)
        print(f.get_face_box(w, h))
        head = f.get_face_box(w, h)
    elapsed = time.time() - t
#BodyPart:0-(0.52, 0.18) score=0.95
    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    try:
        import matplotlib.pyplot as plt
        fig, a = plt.subplots() 
        a.set_title('Result')
        circle1 = plt.Circle((head['x'], head['y']), 20, color='r')
        a.add_artist(circle1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
    except Exception as e:
        logger.warning('matplitlib error, %s' % e)
        cv2.imshow('result', image)
        cv2.waitKey()
