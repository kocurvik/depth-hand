""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import cv2
import os
import sys
import numpy as np
import re
from collections import namedtuple
from colour import Color
import pyzed.sl as sl



class caminfo_ir:
    image_size = (720, 1280)
    region_size = 120
    crop_size = 128  # input image size to models (may changed)
    crop_range = 720  # only operate within this range
    z_range = (100., 1060.)
    anchor_num = 8
    # intrinsic paramters of Intel Realsense SR300
    focal = (475.857, 475.856)
    centre = (310.982, 246.123)
    # joints description
    join_name = [
        'Wrist',
        'TMCP', 'IMCP', 'MMCP', 'RMCP', 'PMCP',
        'TPIP', 'TDIP', 'TTIP',
        'IPIP', 'IDIP', 'ITIP',
        'MPIP', 'MDIP', 'MTIP',
        'RPIP', 'RDIP', 'RTIP',
        'PPIP', 'PDIP', 'PTIP'
    ]
    join_num = 21
    join_type = ('W', 'T', 'I', 'M', 'R', 'P')
    join_color = (
        # Color('cyan'),
        Color('black'),
        Color('magenta'),
        Color('blue'),
        Color('lime'),
        Color('yellow'),
        Color('red')
    )
    join_id = (
        (1, 6, 7, 8),
        (2, 9, 10, 11),
        (3, 12, 13, 14),
        (4, 15, 16, 17),
        (5, 18, 19, 20)
    )
    bone_id = (
        ((0, 1), (1, 6), (6, 11), (11, 16)),
        ((0, 2), (2, 7), (7, 12), (12, 17)),
        ((0, 3), (3, 8), (8, 13), (13, 18)),
        ((0, 4), (4, 9), (9, 14), (14, 19)),
        ((0, 5), (5, 10), (10, 15), (15, 20))
    )
    bbox_color = Color('orange')

    def __init__():
        pass


CamFrames = namedtuple("CamFrames", "depth, color, extra")


class DummyCamFrame:
    def __init__(self):
        self.caminfo = caminfo_ir
        # for test purpose
        self.test_depth = np.zeros(self.caminfo.image_size, dtype=np.uint16)
        self.test_depth[10:20, 10:20] = 240

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    def provide(self):
        return CamFrames(self.test_depth, self.test_depth, self.test_depth)


class FetchHands17:
    def __init__(self, args):
        if args is None:
            raise ValueError('need to provide valid args')
        self.caminfo = args.data_inst
        args.model_inst.check_dir(args.data_inst, args)
        self.depth_image, self.cube = \
            args.model_inst.fetch_random(args)

    def smooth_data(self, scale=5):
        import cv2
        return cv2.bilateralFilter(
            self.depth_image.astype(np.float32),
            5, 30, 30)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    def provide(self):
        return CamFrames(self.depth_image, self.depth_image, self.cube)


class FileStreamer:
    def __init__(self, args):
        if args is None:
            raise ValueError('need to provide valid args')
        self.caminfo = caminfo_ir
        self.args = args
        outdir = args.stream_dir
        print('reading path: ', outdir)
        filelist = [f for f in os.listdir(outdir) if re.match(r'image_D(\d+)\.png', f)]
        if 0 == len(filelist):
            raise ValueError('no stream data found!')
        filelist.sort(key=lambda f: int(args.data_io.imagename2index(f)))
        self.filelist = [
            os.path.join(outdir, f) for f in filelist]
        self.clid = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    def provide(self):
        if self.clid >= len(self.filelist):
            return None
        # print('reading: {}'.format(self.filelist[self.clid]))
        depth_image = self.args.data_io.read_image(
            self.filelist[self.clid]
        )
        self.clid += 1
        return CamFrames(depth_image, depth_image, None)


class ZedCam:
    def __init__(self, args):
        self.caminfo = caminfo_ir

        self.zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_QUALITY  # Use PERFORMANCE depth mode
        init_params.coordinate_units = sl.UNIT.UNIT_MILLIMETER  # Use milliliter units (for depth measurements)

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        self.caminfo.image_size = (self.zed.get_resolution().height, self.zed.get_resolution().width)
        camera_information = self.zed.get_camera_information()
        camera_calib = camera_information.calibration_parameters
        self.caminfo.centre = (camera_calib.left_cam.cx, camera_calib.left_cam.cy)
        self.caminfo.focal = (camera_calib.left_cam.fx, camera_calib.left_cam.fy)

        # self.zed.set_depth_max_range_value(2000)
        self.caminfo.z_range = (self.zed.get_depth_min_range_value(), self.zed.get_depth_max_range_value())

        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.sensing_mode = sl.SENSING_MODE.SENSING_MODE_FILL  # Use STANDARD sensing mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.zed.close()
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    def provide(self):
        image = sl.Mat()
        depth = sl.Mat()
        # confidence = sl.Mat()
        # point_cloud = sl.Mat()

        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.zed.retrieve_image(image, sl.VIEW.VIEW_LEFT) #, type=sl.MEM.MEM_GPU)
            # Retrieve depth map. Depth is aligned on the left image
            self.zed.retrieve_measure(depth, sl.MEASURE.MEASURE_DEPTH) #, type=sl.MEM.MEM_GPU)

        color_image = image.get_data()[:,:,2::-1]
        depth_image = depth.get_data()

        # cv2.imshow("color", color_image)
        # cv2.imshow("depth", depth_image/np.max(depth_image))
        # cv2.waitKey(0)

        # depth_image = np.flip(depth_image, 1)
        # color_image = np.flip(color_image, 1)

        # Remove background - Set to grey
        grey_color = 159
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))
        bg_removed = np.where(
            (depth_image_3d > self.caminfo.z_range[1]) | (depth_image_3d <= 0),
            grey_color, color_image)
        np.clip(
            depth_image,
            self.caminfo.z_range[0], self.caminfo.z_range[1],
            out=depth_image)

        return CamFrames(depth_image.astype(np.uint16), bg_removed, bg_removed)
