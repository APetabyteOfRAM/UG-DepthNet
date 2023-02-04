import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from scipy.spatial.transform import Rotation
from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results
import os
import pybullet
import pybullet_data
from pyquaternion import Quaternion
import serial

import matplotlib.pyplot as pyplot
import numpy
import sys
import time
import os
import pybullet
import pybullet_data


logging.basicConfig(level=logging.INFO)



def calculate_6d_transform_grasp(left, center, right):
    #calculate the 6d transform of the grasp
    a = (left[0], left[1], left[2])
    b = (right[0], right[1], right[2])
    vec1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    vec2 = (center[0], center[1], center[2])

    
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    # return rotation_matrix
    q8d = Quaternion(matrix=rotation_matrix)
    return q8d, center
    

def get_translation(hand_orientation_quat, grasp_position, k_distance, x_offset, y_offset):
    # Rotation matrix
    R = Rotation.from_quat(hand_orientation_quat)

    #grasp poisiton always at origin
    px,py,pz = grasp_position[0], grasp_position[1], grasp_position[2]
    p = np.array([px, py, pz])

    # Distance between the vector and the point
    distance = k_distance
    
    vec_final = R.as_rotvec() * distance + p

    # Planar translation along the new translational plane
    dx, dy = x_offset,y_offset

    # Translation vector
    # Create a unit vector perpendicular to the original vector
    unit_vector = np.cross(vec_final, [0, 0, 1])
    unit_vector = unit_vector / np.linalg.norm(unit_vector)
    # Translate the original vector by dx and dy in the direction of the unit vector
    final_vector = vec_final + np.array([dx*unit_vector[0], dy*unit_vector[1], 0])
    # return vec_final
    return final_vector


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='trained-models\cornell-custom\epoch_30_iou_0.97',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    
    args = parse_args()

    #choose connection method: GUI, DIRECT, SHARED_MEMORY
    pybullet.connect(pybullet.GUI)
    pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, -1)
    #load URDF, given a relative or absolute file+path

    # block_urdf_path = "C:/Users/ram15/AppData/Roaming/Python/Python38/site-packages/pybullet_data/block.urdf"
    grasp_rectangle = pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "block.urdf"))

    orientation_initial = pybullet.getQuaternionFromEuler([0, 0, 0])
    
    grasp_rectangle_cid = pybullet.createConstraint(grasp_rectangle, -1, -1, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                    [0, 0, 0], orientation_initial)

    # hand = pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "r2d2.urdf"))

    # hand_cid = pybullet.createConstraint(hand, -1, -1, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
    #                           [0, 0, 0], orientation_initial)

    pybullet.setGravity(0, 0, 0)

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id=218622277762)
    cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    # device = get_device(args.force_cpu)
    device = get_device(0)
    # dev = "cuda:0"
    # device = torch.device(dev)
    
    # ser = serial.Serial(
    #     port='COM12',\
    #     baudrate=115200,\
    #     parity=serial.PARITY_NONE,\
    #     stopbits=serial.STOPBITS_ONE,\
    #     bytesize=serial.EIGHTBITS,\
    #         timeout=0)

    try:
        

        stream = []
        quat_dict = {}
        acc_dict = {}
        calibration_dict = {}

        forceyforce = 1000000000000
         
        while True:
            
            
                     
            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            depth_frame = image_bundle['aligned_depth_frame']
            depth_frame_unaligned = image_bundle['depth_frame']
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
            with torch.no_grad():
                
                xc = x.to(device)
                # net.set_params(device="cuda:0")
                pred = net.predict(xc)



                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                center, left, right = plot_results(fig=None,
                             rgb_img=cam_data.get_rgb(rgb, False),
                             depth_img=np.squeeze(cam_data.get_depth(depth)),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             no_grasps=args.n_grasps,
                             grasp_width_img=width_img,
                             depth_frame = depth_frame,
                             camera = cam,
                             depth_frame_unaligned = depth_frame_unaligned  )
                
                if center is not None and left is not None and right is not None:
                   
                    # print(["{:.1f}".format(i * 100) for i in left],
                    #   ["{:.1f}".format(i * 100) for i in center],
                    #   ["{:.1f}".format(i * 100) for i in right])
                   
                    orientation_grasp_q8d, position_depth_cam = calculate_6d_transform_grasp(left, center, right)
                    grasp_quaternion = [i for i in orientation_grasp_q8d.elements]
                    pybullet.changeConstraint(grasp_rectangle_cid, [0,0,0], grasp_quaternion, maxForce=forceyforce)
                    pybullet.stepSimulation()
                    
                    
                    # while not (len(quat_dict) == 4 and len(acc_dict) == 3 and len(calibration_dict) == 4):
                    #     if ser.in_waiting > 0:
                    #         input_data=ser.read().strip().decode("utf-8")
                    #         # print(input_data)
                    #         stream.append(input_data)

                    #         if input_data == "q":
                    #             #quaternion
                    #             if "W" in stream:
                    #                 quat_dict["w"] = float(''.join(stream[stream.index("W") + 3:stream.index("X")]))
                    #                 quat_dict["x"] = float(''.join(stream[stream.index("X") + 3:stream.index("Y")]))
                    #                 quat_dict["y"] = float(''.join(stream[stream.index("Y") + 3:stream.index("Z")]))
                    #                 quat_dict["z"] = float(''.join(stream[stream.index("Z") + 3:stream.index("q")]))

                    #             stream = []
                                
                    #         if input_data == "a":
                    #             #linear acceelration with gravity removed
                    #             if "X" in stream:
                    #                 acc_dict["x"] = float(''.join(stream[stream.index("X") + 3:stream.index("Y")]))
                    #                 acc_dict["y"] = float(''.join(stream[stream.index("Y") + 3:stream.index("Z")]))
                    #                 acc_dict["z"] = float(''.join(stream[stream.index("Z") + 3:stream.index("a")]))

                    #             stream = []
                            
                    #         if input_data == "c":
                    #             if "S" in stream:
                    #                 calibration_dict["System"] = int(''.join(stream[stream.index("S") + 2:stream.index("G")]))
                    #                 calibration_dict["Gyro"] = int(''.join(stream[stream.index("G") + 2:stream.index("A")]))
                    #                 calibration_dict["Acceleration"] = int(''.join(stream[stream.index("A") + 2:stream.index("M")]))
                    #                 calibration_dict["Mag"] = int(''.join(stream[stream.index("M") + 2:stream.index("c")]))
                    #             stream = []
                            
                            
                            
                    # if len(quat_dict) == 4 and len(acc_dict) == 3 and len(calibration_dict) == 4:
                    #     orientation_hand_quat = [quat_dict["w"], quat_dict["x"], quat_dict["y"], quat_dict["z"]]
                  
                    #     position_hand = get_translation(orientation_hand_quat, [0,0,0], position_depth_cam[2], position_depth_cam[0], position_depth_cam[1])
                    #     pybullet.changeConstraint(hand_cid, position_hand, orientation_hand_quat , maxForce=forceyforce)
                    #     pybullet.stepSimulation()
                    #     # print(quat_dict)
                    #     # print(acc_dict)
                    #     print(calibration_dict)
                    #     # print(orientation_hand)
                    
                    #     stream = []
                    #     quat_dict = {}
                    #     acc_dict = {}
                    #     calibration_dict = {}
                        
                

    finally:
        print("exited")
        # ser.close()
        # save_results(
        #     rgb_img=cam_data.get_rgb(rgb, False),
        #     depth_img=np.squeeze(cam_data.get_depth(depth)),
        #     grasp_q_img=q_img,
        #     grasp_angle_img=ang_img,
        #     no_grasps=args.n_grasps,
        #     grasp_width_img=width_img
        # )
