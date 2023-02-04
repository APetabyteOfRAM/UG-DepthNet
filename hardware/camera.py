import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import cv2
logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=90):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None
        self.depth_intrinsic = None
        self.color_intrinsic = None
        self.depth_to_color_extrinsic = None
        self.color_to_depth_extrinsic = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        print(self.fps)
        config = rs.config()
        # config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        
        
            
        self.depth_intrinsic = cfg.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.color_intrinsic = cfg.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.depth_to_color_extrinsic =  cfg.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(cfg.get_stream(rs.stream.color))
        self.color_to_depth_extrinsic =  cfg.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(cfg.get_stream(rs.stream.depth))

    ### Project Color Pixel coordinate to Depth Pixel coordinate
    def ProjectColorPixeltoDepthPixel(self,depth_frame,depth_min, depth_max, color_pixel):
        
        depth_scale = self.scale
        depth_intrinsic = self.depth_intrinsic
        color_intrinsic = self.color_intrinsic
        depth_to_color_extrinsic = self.depth_to_color_extrinsic
        color_to_depth_extrinsic = self.color_to_depth_extrinsic
        

        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), depth_scale, 
                        depth_min, depth_max, depth_intrinsic, color_intrinsic, 
                        depth_to_color_extrinsic, color_to_depth_extrinsic, 
                        color_pixel)
        
        return depth_pixel

    ### Deproject Depth Pixel coordinate to Depth Point coordinate
    def DeProjectDepthPixeltoDepthPoint(self,depth_frame, x_depth_pixel, y_depth_pixel):
        depth_intrinsic = self.depth_intrinsic
        
        depth = depth_frame.get_distance(int(x_depth_pixel), int(y_depth_pixel))

        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsic, [int(x_depth_pixel), int(y_depth_pixel)], depth)
        
        return depth, depth_point

    def get_image_bundle(self):
        
        
        
        
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.expand_dims(depth_image, axis=2)

        depth_frame = frames.get_depth_frame()

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
            'aligned_depth_frame': aligned_depth_frame,
            'depth_frame': depth_frame
        }
    

    def plot_image_bundle(self):
        images = self.get_image_bundle()
        
        rgb = images['rgb']
        # depth = images['aligned_depth']
        depth = images['depth_frame']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        fig.show()
        while True:
            images = self.get_image_bundle()
        
            rgb = images['rgb']
            # depth = images['aligned_depth']
            depth = images['depth_frame']
            ax[0, 0].imshow(rgb)
            ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
            plt.pause(0.01)
            fig.show()
            
    def plot_with_opencv(self):
        # Get frameset of color and depth
        images = self.get_image_bundle()
        
        rgb = images['rgb']
        # depth = images['aligned_depth']
        depth_image = images['aligned_depth']
        
        
        color_image = rgb
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / self.scale
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()

        

if __name__ == '__main__':
    cam = RealSenseCamera(device_id=218622277762)
    cam.connect()
    
    # cam.plot_image_bundle()
    while True:
        cam.plot_with_opencv()
