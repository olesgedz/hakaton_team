# """
# This serve as our base openGL class.
# """

# import numpy as np
# import pyqtgraph.opengl as gl
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui
# import sys

# import cv2
# import time
# import os

# from estimator import TfPoseEstimator
# from networks import get_graph_path, model_wh
# from lifting.prob_model import Prob3dPose
# import common


# class Terrain(object):
#     def __init__(self):
#         """
#         Initialize the graphics window and mesh surface
#         """

#         #setup the view window
#         # self.app = QtGui.QApplication(sys.argv)
#         # self.window = gl.GLViewWidget()
#         # self.window.setWindowTitle('Terrain')
#         # self.window.setGeometry(0, 110, 1920, 1080)
#         # self.window.setCameraPosition(distance=30, elevation=12)
#         # self.window.show()

#         # gx = gl.GLGridItem()
#         # gy = gl.GLGridItem()
#         # gz = gl.GLGridItem()
#         # gx.rotate(90, 0, 1, 0)
#         # gy.rotate(90, 1, 0, 0)
#         # gx.translate(-10, 0, 0)
#         # gy.translate(0, -10, 0)
#         # gz.translate(0, 0, -10)
#         # self.window.addItem(gx)
#         # self.window.addItem(gy)
#         # self.window.addItem(gz)

#         model = 'mobilenet_thin_432x368'
#         camera = 0
#         w, h = model_wh(model)
#         print(get_graph_path('mobilenet_thin_432x368'))

#         self.e = TfPoseEstimator(get_graph_path('mobilenet_thin_432x368'), target_size=(w, h)) #error
#         self.cam = cv2.VideoCapture(camera)

      

#         self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

#         while True:
#             ret_val, image = self.cam.read()
#             keypoints  = self.mesh(image)
#             print(keypoints)
#             cv2.imshow('tf-pose-estimation result', image)
#             if cv2.waitKey(1) == 27: 
#                 break
#         cv2.destroyAllWindows()
#         # self.window.addItem(self.points)

#     def mesh(self, image):
#         image_h, image_w = image.shape[:2]
#         width = 432
#         height = 368
#         pose_2d_mpiis = []
#         visibilities = []

#         humans = self.e.inference(image, scales=[None])

#         for human in humans:
#             pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
#             pose_2d_mpiis.append(
#                 [(int(x * width + 0.5), int(y * height + 0.5)) for x, y in pose_2d_mpii]
#             )
#             visibilities.append(visibility)

#         pose_2d_mpiis = np.array(pose_2d_mpiis)
#         visibilities = np.array(visibilities)
#         try:
#             transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
#             pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
#             for i, single_3d in enumerate(pose_3d):
#                 plot_pose(single_3d)
#                 avrg = []
#                 number = 0
#                 for i, point in single_3d[0]:
#                     avrg += point[2]
#                     number+=1
#             print("\n new \n")
#             print(avrg/num)   
#             plt.show()
#             keypoints = pose_3d[0].transpose()
#             return keypoints / 80
#         except:
#             pass
#             return 0

#     def update(self):
#         """
#         update the mesh and shift the noise each time
#         """
#         ret_val, image = self.cam.read()
#         try:
#             keypoints = self.mesh(image)
#         except AssertionError:
#             print('body not in image')
#         else:
#             self.points.setData(pos=keypoints)

#     def start(self):
#         """
#         get the graphics window open and setup
#         """
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()

#     def animation(self, frametime=10):
#         """
#         calls the update method to run in a loop
#         """
#         timer = QtCore.QTimer()
#         timer.timeout.connect(self.update)
#         timer.start(frametime)
#         self.start()


# if __name__ == '__main__':
#     #os.chdir('..')
#     t = Terrain()
#     #t.animation()

# # """
# # This serve as our base openGL class.
# # """

# # import numpy as np
# # import pyqtgraph.opengl as gl
# # import pyqtgraph as pg
# # from pyqtgraph.Qt import QtCore, QtGui
# # import sys

# # import cv2
# # import time
# # import os
# # from lifting.prob_model import Prob3dPose
# # from lifting.draw import plot_pose
# # import matplotlib.pyplot as plt
# # from estimator import TfPoseEstimator
# # from networks import get_graph_path, model_wh
# # from lifting.prob_model import Prob3dPose
# # import common
# # import streamlink

# # class Terrain(object):
# #     def __init__(self):
# #         """
# #         Initialize the graphics window and mesh surface
# #         """

# #         #setup the view window
# #         # self.app = QtGui.QApplication(sys.argv)
# #         # self.window = gl.GLViewWidget()
# #         # self.window.setWindowTitle('Terrain')
# #         # self.window.setGeometry(0, 110, 1920, 1080)
# #         # self.window.setCameraPosition(distance=30, elevation=12)
# #         # self.window.show()

# #         # gx = gl.GLGridItem()
# #         # gy = gl.GLGridItem()
# #         # gz = gl.GLGridItem()
# #         # gx.rotate(90, 0, 1, 0)
# #         # gy.rotate(90, 1, 0, 0)
# #         # gx.translate(-10, 0, 0)
# #         # gy.translate(0, -10, 0)
# #         # gz.translate(0, 0, -10)
# #         # self.window.addItem(gx)
# #         # self.window.addItem(gy)
# #         # self.window.addItem(gz)
# #         streams = streamlink.streams('https://www.youtube.com/watch?v=1EiC9bvVGnk')#https://www.twitch.tv/blondynkitezgraja')
# #     # #print(streams)
# #         url = streams['360p'].url
# #         model = 'mobilenet_thin_432x368'
# #         camera = 0
# #         self.w, self.h = model_wh(model)
# #         print(get_graph_path('mobilenet_thin_432x368'))

# #         self.e = TfPoseEstimator(get_graph_path('mobilenet_thin_432x368'), target_size=(self.w, self.h)) #error
# #         self.cam = cv2.VideoCapture(camera)
# #         ret_val, self.image = self.cam.read()
# #         print(self.image.shape[1])
# #         print(self.image.shape[0])
      

# #         self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

# #         while True:
# #             ret_val, self.image = self.cam.read()
# #             keypoints  = self.mesh()
# #             print(keypoints)

            
# #             if cv2.waitKey(1) == 27: 
# #                 break
# #         cv2.destroyAllWindows()
# #         # self.window.addItem(self.points)

# #     def mesh(self):
# #         image_h, image_w = self.image.shape[:2]
# #         width = 640
# #         height = 480
# #         pose_2d_mpiis = []
# #         visibilities = []
# #         dim = (self.w, self.h)
# #         #self.image = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)
# #         humans = self.e.inference(self.image, scales=[None])
# #         self.image = TfPoseEstimator.draw_humans(self.image, humans)
# #         cv2.imshow('tf-pose-estimation result', self.image)
# #         for human in humans:
# #             pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
# #             pose_2d_mpiis.append(
# #                 [(int(x * self.w + 0.5), int(y * self.h + 0.5)) for x, y in pose_2d_mpii]
# #             )
# #             visibilities.append(visibility)
# #         pose_2d_mpiis = np.array(pose_2d_mpiis)
# #         visibilities = np.array(visibilities)
# #         poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')
# #         try:
# #             transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
# #             pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
# #             for i, single_3d in enumerate(pose_3d):
# #                 plot_pose(single_3d)
# #                 avrg = []
# #                 number = 0
# #                 for i, point in single_3d[0]:
# #                     avrg += point[2]
# #                     number+=1
# #                 print("\n new \n")
# #                 print(avrg/num)   
# #             plt.show()
# #             keypoints = pose_3d[0].transpose()
# #             return keypoints / 80
# #         except:
# #             pass
# #             return 0

# #     def update(self):
# #         """
# #         update the mesh and shift the noise each time
# #         """
# #         ret_val, image = self.cam.read()
# #         try:
# #             keypoints = self.mesh(image)
# #         except AssertionError:
# #             print('body not in image')
# #         else:
# #             self.points.setData(pos=keypoints)

# #     def start(self):
# #         """
# #         get the graphics window open and setup
# #         """
# #         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
# #             QtGui.QApplication.instance().exec_()

# #     def animation(self, frametime=10):
# #         """
# #         calls the update method to run in a loop
# #         """
# #         timer = QtCore.QTimer()
# #         timer.timeout.connect(self.update)
# #         timer.start(frametime)
# #         self.start()


# # if __name__ == '__main__':
# #     #os.chdir('..')
# #     t = Terrain()
# #     #t.animation()

"""
This serve as our base openGL class.
"""

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys

import cv2
import time
import os
import matplotlib.pyplot as plt
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from lifting.prob_model import Prob3dPose
import common


class Terrain(object):
    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """

        #setup the view window
        # self.app = QtGui.QApplication(sys.argv)
        # self.window = gl.GLViewWidget()
        # self.window.setWindowTitle('Terrain')
        # self.window.setGeometry(0, 110, 1920, 1080)
        # self.window.setCameraPosition(distance=30, elevation=12)
        # self.window.show()

        # gx = gl.GLGridItem()
        # gy = gl.GLGridItem()
        # gz = gl.GLGridItem()
        # gx.rotate(90, 0, 1, 0)
        # gy.rotate(90, 1, 0, 0)
        # gx.translate(-10, 0, 0)
        # gy.translate(0, -10, 0)
        # gz.translate(0, 0, -10)
        # self.window.addItem(gx)
        # self.window.addItem(gy)
        # self.window.addItem(gz)

        model = 'mobilenet_thin_432x368'
        camera = 0
        w, h = model_wh(model)
        print(get_graph_path('mobilenet_thin_432x368'))

        self.e = TfPoseEstimator(get_graph_path('mobilenet_thin_432x368'), target_size=(w, h)) #error
        self.cam = cv2.VideoCapture(camera)

      

        self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

        while True:
            ret_val, self.image = self.cam.read()
            pose_3d = self.mesh()
            keypoints  = pose_3d[0].transpose()
            keypoints = keypoints / 80
            print(keypoints)
            for i, single_3d in enumerate(pose_3d):
                #plot_pose(single_3d)
                avrg = []
                number = 0
                for point in single_3d:
                    avrg += point[2]
                    print(point[2])
                    number+=1   
            # print("\n new \n")
            print(avrg)   
            plt.show()
            if cv2.waitKey(1) == 27: 
                break
        cv2.destroyAllWindows()
        #self.window.addItem(self.points)

    def mesh(self):
        image_h, image_w = self.image.shape[:2]
        width = 640
        height = 480
        pose_2d_mpiis = []
        visibilities = []

        humans = self.e.inference(self.image, scales=[None])
        self.image = TfPoseEstimator.draw_humans(self.image, humans)
        cv2.imshow('tf-pose-estimation result', self.image)
        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append(
                [(int(x * width + 0.5), int(y * height + 0.5)) for x, y in pose_2d_mpii]
            )
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)
        visibilities = np.array(visibilities)
        try:
            transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
            pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        
            return  pose_3d
        except:
            pass
            return 0

    def update(self):
        """
        update the mesh and shift the noise each time
        """
        ret_val, image = self.cam.read()
        try:
            keypoints = self.mesh(image)
        except AssertionError:
            print('body not in image')
        else:
            self.points.setData(pos=keypoints)

    def start(self):
        """
        get the graphics window open and setup
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def animation(self, frametime=10):
        """
        calls the update method to run in a loop
        """
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()


if __name__ == '__main__':
    #os.chdir('..')
    t = Terrain()
    #t.animation()
