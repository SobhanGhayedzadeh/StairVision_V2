import open3d as o3d
import numpy as np

class LivePCDVisualizer:
    def __init__(self, window_name="Live Point Cloud (RViz Style)", width=800, height=600):
        """Initializes the non-blocking Open3D visualizer."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)
        
        # We maintain a single legacy point cloud object and update its data
        self.pcd_legacy = o3d.geometry.PointCloud()
        self.is_initialized = False

    def update(self, pcd_tensor):
        """Updates the visualizer with the new point cloud frame."""
        # 1. Move tensor data back to the CPU
        points_cpu = pcd_tensor.point["positions"].to(o3d.core.Device("CPU:0")).numpy()
        self.pcd_legacy.points = o3d.utility.Vector3dVector(points_cpu)
        
        # 2. Handle colors if they exist
        if "colors" in pcd_tensor.point:
            colors_cpu = pcd_tensor.point["colors"].to(o3d.core.Device("CPU:0")).numpy()
            if colors_cpu.dtype == np.uint8:
                colors_cpu = colors_cpu.astype(np.float32) / 255.0
            self.pcd_legacy.colors = o3d.utility.Vector3dVector(colors_cpu)

        # 3. Add geometry on the first frame, update it on subsequent frames
        if not self.is_initialized:
            self.vis.add_geometry(self.pcd_legacy)
            self.is_initialized = True
        else:
            self.vis.update_geometry(self.pcd_legacy)
        
        # 4. Process window events to keep the UI responsive
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        """Destroys the Open3D window cleanly."""
        self.vis.destroy_window()