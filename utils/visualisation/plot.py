import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps
import cv2
                         #x,y
def get_3d_point(camera, color_pixel, depth_frame):
            # distance = depth_frame.get_distance(*gs[0].center)
    depth_min = 0.02 #meter
    depth_max = 0.55 #meter
    ### Get depth pixel from color pixel
    depth_pixel = camera.ProjectColorPixeltoDepthPixel(depth_frame, depth_min, depth_max, color_pixel)

    x_depth_pixel, y_depth_pixel = depth_pixel
    
    # print("depth_pixel: ", depth_pixel)
    
    if depth_pixel != [-1,-1]:

        ### Get depth points from depth pixel
        depth, depth_point = camera.DeProjectDepthPixeltoDepthPoint(depth_frame, x_depth_pixel, y_depth_pixel)

        x_depth_point, y_depth_point, z_depth_point = depth_point 
        
        return (x_depth_point, y_depth_point, z_depth_point)
    
    else: return None
    

warnings.filterwarnings("ignore")
import pyrealsense2 as rs
def plot_results(
        fig,
        rgb_img,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img=None,
        depth_frame=None,
        camera = None,
        depth_frame_unaligned = None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    
    if len(gs) > 0:
             
        ############################# 
        
        grasp = gs[0] 
        
        xo = np.cos(grasp.angle)
        yo = np.sin(grasp.angle)


        y1 = grasp.center[0] + grasp.length / 2 * yo
        x1 = grasp.center[1] - grasp.length / 2 * xo
        
        y2 = grasp.center[0] - grasp.length / 2 * yo
        x2 = grasp.center[1] + grasp.length / 2 * xo
        
        pts = np.array([
                [x1 - grasp.width / 2 * yo, y1 - grasp.width / 2 * xo ],
                [x2 - grasp.width / 2 * yo, y2 - grasp.width / 2 * xo ],
                [x2 + grasp.width / 2 * yo, y2 + grasp.width / 2 * xo ],
                [x1 + grasp.width / 2 * yo, y1 + grasp.width / 2 * xo ],
            ], np.int32)
        
        midptsx = [(pts[0][0] + pts[1][0])/2, (pts[1][0] + pts[2][0])/2, (pts[2][0] + pts[3][0])/2, (pts[3][0] + pts[0][0])/2]
        midptsy = [(pts[0][1] + pts[1][1])/2, (pts[1][1] + pts[2][1])/2, (pts[2][1] + pts[3][1])/2, (pts[3][1] + pts[0][1])/2]
        midpts = zip(midptsx, midptsy)
        midpoints_actual = set()
        for midpt in midpts:
            for pt in pts:
                if np.sqrt((midpt[0] - pt[0]) ** 2 + (midpt[1] - pt[1]) ** 2 ) - (grasp.width//2) < 2:
                    midpoints_actual.add(midpt)
        # print(grasp.center[1], grasp.center[0])
        # print(midpoints_actual)
        
        first = [int(i) for i in midpoints_actual.pop()]
        second = [int(i) for i in midpoints_actual.pop()]
                  
        center = get_3d_point(camera, (grasp.center[1] + 208, grasp.center[0] + 128), depth_frame_unaligned)
        left =  get_3d_point(camera, (first[0] + 208, first[1] + 128), depth_frame_unaligned)
        right = get_3d_point(camera, (second[0] + 208, second[1] + 128), depth_frame_unaligned)
        
                # pts = pts.reshape((-1, 1, 2))
        
        image = cv2.polylines(rgb_img, [pts],
                    True, (255, 200, 150), 2)
        
        image = cv2.line(image, (first[0   ], first[1]), (second[0], second[1]), (255, 255, 0), 2)

      
        image = cv2.circle(image, (grasp.center[1], grasp.center[0]), 3, (155, 0, 255), -1)
        image = cv2.circle(image, (first[0], first[1]), 3, (155, 0, 255), -1)
        image = cv2.circle(image, (second[0], second[1]), 3, (155, 0, 255), -1)

            
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.resizeWindow('RealSense', 848,480)
        cv2.imshow('RealSense', image)
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
        
        return (center, left, right)
    
    return None, None, None
  
