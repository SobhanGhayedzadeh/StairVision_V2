import open3d as o3d
import numpy as np
import torch
from torch.utils import dlpack
import math 

class PointCloudHandler:
    def __init__(self, K_matrix_np, device_str="CUDA:0", depth_scale=1000.0, depth_max=5.0):
        """
        Initializes the PointCloud Handler. Loads the intrinsic matrix onto the target device.
        """
        self.device = o3d.core.Device(device_str)
        self.K = o3d.core.Tensor(K_matrix_np, device=self.device)
        self.depth_scale = float(depth_scale)
        self.depth_max = float(depth_max)

    def pcd_from_rgbd_cuda(self, color_np: np.ndarray, depth_np: np.ndarray):
        """Copies raw frames to the GPU and creates a point cloud in one hop."""
        color_t = o3d.core.Tensor(color_np, o3d.core.Dtype.UInt8, self.device)
        depth_t = o3d.core.Tensor(depth_np, o3d.core.Dtype.UInt16, self.device)
        
        color_img = o3d.t.geometry.Image(color_t)
        depth_img = o3d.t.geometry.Image(depth_t)
        rgbd = o3d.t.geometry.RGBDImage(color_img, depth_img)

        pcd_t = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd, 
            intrinsics=self.K,
            depth_scale=self.depth_scale,
            depth_max=self.depth_max
        )

        return pcd_t

    def downsample_and_crop(self, pcd, voxel_size: float, minmax_Y_to_crop: float, tile_dim: float):
        """Applies voxel downsampling and spatial bounding box masking."""

        pcd = pcd.voxel_down_sample(voxel_size=min(voxel_size,tile_dim))
        # pcd = pcd.uniform_down_sample(every_k_points=50)
        
        pos = pcd.point["positions"]
        mask = (
            (pos[:, 2] > 0.3) & (pos[:, 2] < 3.5) &
            (pos[:, 0] > -2.5) & (pos[:, 0] < 2.5) &
            (pos[:, 1] > -minmax_Y_to_crop) & (pos[:, 1] < minmax_Y_to_crop)
        )
        
        return pcd.select_by_mask(mask)

    def rotate(self, pcd, R1, R2):
        """Rotates the point cloud using pre-calculated Open3D tensor matrices."""
        dtype = pcd.point["positions"].dtype
        center = o3d.core.Tensor([0, 0, 0], dtype=dtype, device=self.device)
        
        pcd = pcd.rotate(R1, center=center)
        pcd = pcd.rotate(R2, center=center)
        return pcd

    def translate_to_origin(self, pcd):
        """Translates the point cloud so its lowest bounds touch (0,0,0)."""
        pos = pcd.point["positions"]
        dtype = pos.dtype
        
        xyz_to_translate = (-1.0) * o3d.core.Tensor(
            [float(pos[:, 0].min().item()), 
             float(pos[:, 1].min().item()), 
             float(pos[:, 2].min().item())], 
            dtype=dtype, 
            device=self.device
        )
        
        pcd = pcd.translate(xyz_to_translate)
        
        # We return xyz_to_translate so you can reverse this translation later for the UI projection
        return pcd, xyz_to_translate 

    @staticmethod
    def o3d_to_torch(o3d_tensor: o3d.core.Tensor) -> torch.Tensor:
        """Zero-copy view of an Open3D tensor as a PyTorch tensor."""
        return dlpack.from_dlpack(o3d_tensor.to_dlpack())

    @torch.inference_mode()
    def build_height_map_gpu(self, pcd, tile_dim: float):
        """
        GPU version of the Numba hight_map_builder.
        Takes a processed Open3D point cloud and returns a 2D PyTorch height map.
        """
        points_o3d = pcd.point["positions"].clone()
        pts = self.o3d_to_torch(points_o3d)

        # 1) Convert Z to "height from ground"
        max_z = pts[:, 2].max()
        pts[:, 2] = max_z - pts[:, 2]

        # 2) Swap X <-> Y so X is "row" and Y is "col"
        pts[:, :2] = pts[:, [1, 0]]

        # 3) Compute integer tile indices
        tile_idx = torch.floor(pts[:, 0] / tile_dim).to(torch.int64)
        tile_idy = torch.floor(pts[:, 1] / tile_dim).to(torch.int64)

        num_tiles_x = int(tile_idx.max().item()) + 1
        num_tiles_y = int(tile_idy.max().item()) + 1

        # 4) Flatten 2-D indices -> 1-D so we can `scatter_add`
        flat_idx = tile_idx * num_tiles_y + tile_idy

        # 5) Accumulate sums and counts
        h_sum = torch.zeros(num_tiles_x * num_tiles_y, dtype=pts.dtype, device=pts.device)
        h_count = torch.zeros_like(h_sum, dtype=torch.int32)

        h_sum.scatter_add_(0, flat_idx, pts[:, 2])
        h_count.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.int32))

        # 6) Compute means only where we have at least one sample
        valid = h_count > 0
        h_sum[valid] = h_sum[valid] / h_count[valid].to(h_sum.dtype)

        # 7) Reshape back to 2-D height map
        return h_sum.view(num_tiles_x, num_tiles_y)
    
    def visualize(self, pcd, window_name="Point Cloud Visualization"):
        """
        Safely visualizes an Open3D tensor point cloud.
        Moves the data to the CPU and converts it to a legacy point cloud for the Open3D viewer.
        """
        # Move positions to CPU and convert to numpy
        points_cpu = pcd.point["positions"].to(o3d.core.Device("CPU:0")).numpy()
        
        # Create a legacy Open3D point cloud
        legacy_pcd = o3d.geometry.PointCloud()
        legacy_pcd.points = o3d.utility.Vector3dVector(points_cpu)
        
        # If the point cloud has colors, move them over too
        if "colors" in pcd.point:
            colors_cpu = pcd.point["colors"].to(o3d.core.Device("CPU:0")).numpy()
            # Open3D legacy viewer expects colors as floats between 0.0 and 1.0
            if colors_cpu.dtype == np.uint8:
                colors_cpu = colors_cpu.astype(np.float32) / 255.0
            legacy_pcd.colors = o3d.utility.Vector3dVector(colors_cpu)
            
        # Launch the viewer (execution will pause here until you close the window)
        o3d.visualization.draw_geometries([legacy_pcd], window_name=window_name)

    def crop_with_open3d_builtin(self, corners_2d, pcd):
        """Converts 4 YOLO corners into an Open3D Tensor OBB and crops the point cloud."""
        # Force the input into a 4x2 matrix
        corners_2d = np.asarray(corners_2d).reshape(4, 2) 
        
        # 1. Calculate the Center (Mean of the 4 corners)
        center_x = np.mean(corners_2d[:, 0])
        center_y = np.mean(corners_2d[:, 1])
        center_z = 0.0 
        
        # Wrap the list in a NumPy array first
        center_arr = np.array([center_x, center_y, center_z], dtype=np.float32)
        center = o3d.core.Tensor(center_arr, o3d.core.float32, self.device)

        # 2. Calculate the Extent (Width and Length via distance formula)
        width = np.linalg.norm(corners_2d[0] - corners_2d[1])
        length = np.linalg.norm(corners_2d[1] - corners_2d[2])
        height = 10.0 # Arbitrary huge height to capture all stairs
        
        # Wrap the list in a NumPy array first
        extent_arr = np.array([width, length, height], dtype=np.float32)
        extent = o3d.core.Tensor(extent_arr, o3d.core.float32, self.device)

        # 3. Calculate the Rotation Matrix (Angle of the first edge)
        dx = corners_2d[1, 0] - corners_2d[0, 0]
        dy = corners_2d[1, 1] - corners_2d[0, 1]
        angle = math.atan2(dy, dx)
        
        R_np = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle),  math.cos(angle), 0.0],
            [0.0,              0.0,             1.0]
        ], dtype=np.float32)
        rotation = o3d.core.Tensor(R_np, o3d.core.float32, self.device)

        # 4. Create the OBB and Crop
        obb = o3d.t.geometry.OrientedBoundingBox(center, rotation, extent)
        cropped_pcd = pcd.crop(obb)
        
        return cropped_pcd
    
    @torch.inference_mode()
    def filter_by_normal_angle(self, pcd, radius, angle_threshold=30.0):
        """
        Estimates normals and removes points that belong to walls (vertical surfaces).
        Only keeps points whose normal is pointing mostly up or down.
        """
        # 1. Estimate normals on the GPU
        pcd.estimate_normals(radius=radius, max_nn=10)
        
        # 2. Get normals as a zero-copy PyTorch tensor
        normals = self.o3d_to_torch(pcd.point["normals"])
        
        # 3. Define the Z-axis vector (ground normal)
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=normals.device)
        
        # 4. Calculate angles (Open3D normals are already normalized)
        dot_product = torch.clamp(torch.matmul(normals, z_axis), -1.0, 1.0)
        angles_deg = torch.abs(torch.rad2deg(torch.acos(dot_product)))
        
        # 5. Create the mask (< 30 or > 150)
        mask = (angles_deg < angle_threshold) | (angles_deg > (180 - angle_threshold))
        valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        
        # 6. Apply mask back to Open3D
        # (Route indices through CPU numpy to safely match your original syntax)
        idx_o3d = o3d.core.Tensor(valid_idx.cpu().numpy(), dtype=o3d.core.int64, device=self.device)
        filtered_pcd = pcd.select_by_index(idx_o3d)
        
        return filtered_pcd, len(valid_idx)