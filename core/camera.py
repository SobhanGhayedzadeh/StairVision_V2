import pyrealsense2 as rs
import numpy as np
import os

class RealSenseCamera:
    def __init__(self, bag_file=None, repeat=False, width=640, height=480, fps=30):
        """
        Initializes the RealSense pipeline. Switches to offline mode if bag_file is provided.
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        
        self.bag_file = bag_file
        
        if self.bag_file:
            if not os.path.isfile(self.bag_file):
                raise FileNotFoundError(f"Bag file not found: {self.bag_file}")
            self.config.enable_device_from_file(self.bag_file, repeat_playback=repeat)
        else:
            # Online mode: configure standard streams
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
            self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    def start(self):
        """Starts the pipeline and retrieves camera intrinsics and depth scale."""
        self.profile = self.pipeline.start(self.config)
        
        # Get device to extract depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Extract intrinsics for K matrix
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = rs.video_stream_profile(color_stream).get_intrinsics()
        
        self.K_matrix = np.array([
            [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
            [0.0, self.intrinsics.fy, self.intrinsics.ppy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        return self.K_matrix, self.depth_scale

    def get_frames(self):
        """Waits for frames, aligns them, and extracts images and IMU data as numpy arrays."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None
            
        color_image = np.asarray(color_frame.get_data())
        depth_image = np.asarray(depth_frame.get_data())
        
        # IMU extraction
        accel_frame = frames.first(rs.stream.accel)
        gyro_frame  = frames.first(rs.stream.gyro)
        
        accel_array, gyro_array = None, None
        
        if accel_frame:
            a = accel_frame.as_motion_frame().get_motion_data()
            accel_array = np.array([a.x, a.y, a.z], dtype=np.float32)
            
        if gyro_frame:
            g = gyro_frame.as_motion_frame().get_motion_data()
            gyro_array = np.array([g.x, g.y, g.z], dtype=np.float32)
            
        return {
            "color": color_image,
            "depth": depth_image,
            "accel": accel_array,
            "gyro": gyro_array
        }

    def stop(self):
        """Stops the RealSense pipeline."""
        self.pipeline.stop()