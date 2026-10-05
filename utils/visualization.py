import cv2
import numpy as np
import open3d as o3d
import torch
import time
import os

class StairVisualizer:
    def __init__(self, K_matrix_np, width=640, height=480, device="CUDA:0"):
        self.width = width
        self.height = height
        self.device = o3d.core.Device(device)
        self.K_tensor = o3d.core.Tensor(K_matrix_np.astype(np.float64), dtype=o3d.core.float64, device=self.device)

    def extract_and_transform_steps(self, stairs_cloud, p_labels, min_points_threshold, xyz_translate, kf):
        """Filters DBSCAN clusters into individual treads and transforms them to camera frame."""
        unique_labels, counts = np.unique(p_labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        
        # Filter noise (-1) and small clusters
        filtered_labels = [lbl for lbl, cnt in label_counts.items() if cnt >= min_points_threshold and lbl != -1]
        if len(filtered_labels) < 2:
            return []

        # 1. Group points per label
        stairs_points = []
        for label in filtered_labels:
            mask = np.where(p_labels == label)[0]
            mask_tensor = o3d.core.Tensor(mask, dtype=o3d.core.int64, device=self.device)
            stairs_points.append(stairs_cloud.select_by_index(mask_tensor))

        # 2. Translate back from origin
        for stair in stairs_points:
            stair.translate(-xyz_translate)

        # 3. Sort steps by 2D distance to camera origin (closest to furthest)
        distances = [
            float(torch.norm(torch.from_numpy(stair.point["positions"][:, :2].cpu().numpy().astype(np.float32)), dim=1).mean().item())
            for stair in stairs_points
        ]
        sorted_indices = np.argsort(distances)
        stairs_points = [stairs_points[i] for i in sorted_indices]

        # 4. Rotate back to original camera view
        dtype = stairs_cloud.point["positions"].dtype
        R1_inv, R2_inv = kf.get_inverse_rotation_matrices(dtype=dtype, device=self.device)
        center = o3d.core.Tensor([0, 0, 0], dtype=dtype, device=self.device)
        
        for stair in stairs_points:
            stair.rotate(R2_inv, center=center)
            stair.rotate(R1_inv, center=center)

        return stairs_points

    def project_and_overlay(self, color_image, stairs_points):
        """Projects 3D step points into 2D camera coordinates and blends colored contours."""
        stairs_contours = []
        
        for stair in stairs_points:
            stair_rgbd = stair.project_to_rgbd_image(
                width=self.width,
                height=self.height,
                intrinsics=self.K_tensor,
                depth_max=8.0
            )
            
            stair_mask = np.asarray(stair_rgbd.color.cpu())
            stair_mask = cv2.cvtColor(stair_mask, cv2.COLOR_BGR2GRAY)
            nonzero_points = cv2.findNonZero(stair_mask)
            
            if nonzero_points is None or len(nonzero_points) < 10:
                continue

            # Convex hull + morphological smoothing
            hull = cv2.convexHull(nonzero_points)
            contour_img = np.zeros_like(stair_mask)
            cv2.drawContours(contour_img, [hull], -1, 255, thickness=cv2.FILLED)
            
            kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            contour_img = cv2.morphologyEx(contour_img, cv2.MORPH_DILATE, kernel_5, iterations=1)
            contour_img = cv2.morphologyEx(contour_img, cv2.MORPH_ERODE, kernel_5, iterations=1)
            
            kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            contour_img = cv2.morphologyEx(contour_img, cv2.MORPH_ERODE, kernel_3, iterations=2)
            
            stairs_contours.append(contour_img)

        if len(stairs_contours) < 2:
            return color_image, 0

        # Assign unique colors from JET map per step
        num_steps = len(stairs_contours)
        colors = np.arange(0, num_steps, dtype=np.uint8).reshape(1, -1)
        colors = cv2.normalize(colors, None, 0, 240, cv2.NORM_MINMAX)
        colors = cv2.applyColorMap(colors, cv2.COLORMAP_JET)

        output_mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i, contour in enumerate(stairs_contours):
            color = tuple(int(c) for c in colors[0, i])
            cv2.drawContours(output_mask, cv2.findNonZero(contour), -1, color, thickness=cv2.FILLED)

        # Alpha blend onto original RGB image
        mask = cv2.cvtColor(output_mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.drawContours(mask, cv2.findNonZero(mask), -1, 255, thickness=cv2.FILLED)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        result = cv2.bitwise_and(output_mask, mask)
        blended = cv2.addWeighted(color_image, 1.0, result, 1.0, 0.0)
        
        return blended, num_steps

    def draw_info(self, info_img, dt, kf, num_steps=0, status_text="Stairs Not Found", color=(0, 0, 255)):
        """Draws cycle time, IMU angles, step count, and status on the info panel."""
        info_img.fill(0)
        cv2.putText(info_img, "Cycle Time:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, f"{dt:.3f} s", (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(info_img, "Camera Angles:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, f"{180 + kf.angle_x:.1f}, {kf.angle_y:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(info_img, "Steps Count:", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, str(num_steps), (160, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(info_img, status_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return info_img
    

    def save_frame_data(self, color_image, depth_colormap, height_map, bbox_img, overlay_img, info_img, step_count, kf, dt, base_dir="saved_frames"):
        """Saves all visual and textual data of the current frame to a timestamped folder."""
        # Create a unique directory for this specific frame
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_dir, f"frame_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Save all the image windows
        cv2.imwrite(os.path.join(save_dir, "1_rgb.png"), color_image)
        cv2.imwrite(os.path.join(save_dir, "2_depth.png"), depth_colormap)
        cv2.imwrite(os.path.join(save_dir, "3_height_map.png"), height_map)
        cv2.imwrite(os.path.join(save_dir, "4_bbox.png"), bbox_img)
        cv2.imwrite(os.path.join(save_dir, "5_contours.png"), overlay_img)
        cv2.imwrite(os.path.join(save_dir, "6_info_panel.png"), info_img)

        # Save the numerical data to a text file
        txt_path = os.path.join(save_dir, "info.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Cycle Time: {dt:.3f} s\n")
            f.write(f"Camera Angle X (Pitch): {180 + kf.angle_x:.1f} deg\n")
            f.write(f"Camera Angle Y (Roll): {kf.angle_y:.1f} deg\n")
            f.write(f"Detected Steps: {step_count}\n")
        
        print(f"\n[SAVE] Frame data successfully saved to: {save_dir}")