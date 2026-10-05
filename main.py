import argparse
import cv2
import sys
import open3d as o3d
from core.camera import RealSenseCamera
from core.kf1 import KalmanFilter
from core.pointcloud_handler import PointCloudHandler
from utils.Live_Viz import LivePCDVisualizer
from utils.profiler import PerformanceProfiler
from utils.visualization import StairVisualizer
import numpy as np
import time
from ultralytics import YOLO


def main():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Stair Detection v2")
    parser.add_argument("--mode", type=str, default="online", choices=["online", "offline"], 
                        help="Run 'online' with a physical camera or 'offline' with a .bag file.")
    parser.add_argument("--bag", type=str, default="run01.bag", 
                        help="Path to .bag file (only used if --mode offline)")
    parser.add_argument("--repeat", action="store_true", 
                        help="Loop the bag file playback")
    args = parser.parse_args()

    # 2. Initialize the Camera Module
    bag_path = args.bag if args.mode == "offline" else None
    
    print(f"Initializing camera in {args.mode} mode...")
    try:
        camera = RealSenseCamera(bag_file=bag_path, repeat=args.repeat)
        K_matrix, depth_scale = camera.start()
        print("Camera started successfully!")
        print(f"Depth scale: {depth_scale} m/unit")
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        sys.exit(1)
    
    # 3. Initialize Kalman Filter and Timers
    kf = KalmanFilter()
    last_time = time.time()
    info_img = np.zeros((200,430,3),np.uint8)

    # Initialize the PCD-Handler
    pc_handler = PointCloudHandler(K_matrix, device_str="CUDA:0", depth_scale=1/depth_scale)

    # Initialize the live viewer outside the loop
    # live_viewer = LivePCDVisualizer()
    stair_vis = StairVisualizer(K_matrix, width=640, height=480, device="CUDA:0")

    profiler = PerformanceProfiler()

    # Initialize YOLO 
    hmap_img_size = 480
    confidence = 0.3
    model = YOLO("best.pt")
    try:
        model = model.cuda()
    except:
        print("!!!!!!!!!!!!    CUDA IS UNAVAILABLE    !!!!!!!!!!!!")
        sys.exit(1)

    print("YOLO Device:", model.device)
    
    # First call, to avoid freezing in the main loop
    _ = model.predict(np.zeros((hmap_img_size, hmap_img_size, 3), np.uint8), max_det=1, imgsz=hmap_img_size, conf=confidence, iou=0.7, verbose=False)
    print("YOLO predict function started")

    tile_dim = 0.05
    down_sampling_voxel_size=0.04
    min_points_in_obb_area = 0.15/(down_sampling_voxel_size**2)

    # 4. Main Loop
    print("Starting frame stream... Press 'q' or 'Esc' to exit.")
    try:
        while True:
            # Fetch synchronized frames and IMU data
            frames = camera.get_frames()
            
            if frames is None:
                continue 
            
            color_image = frames["color"]
            depth_image = frames["depth"]
            accel_array = frames["accel"]
            gyro_array  = frames["gyro"]
            
            # Apply color map to depth image for visualization (using your alpha scale)
            _alpha = 255.0 / 5000 
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=_alpha, beta=0), 
                cv2.COLORMAP_JET
            )

            # Calculate the current time and time difference
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            start_time = time.perf_counter()

            # Displaying cycle time
            cv2.imshow("Info",info_img)
            info_img=np.zeros((200,430,3),np.uint8)
            info_img=cv2.putText(info_img,"Cycle Time:",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,2)
            info_img=cv2.putText(info_img,str("{0:.3f}".format(dt))+" s",(160,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,2)

            # Calculate angles from accelerometer
            accel_angle_x = np.arctan2(accel_array[1], accel_array[2]) * (180.0 / np.pi)
            accel_angle_y = np.arctan2(-accel_array[0], np.sqrt(accel_array[1]**2 + accel_array[2]**2)) * (180.0 / np.pi)

            # Update the Kalman filter
            kf.update(accel_angle_x, accel_angle_y, gyro_array[0], gyro_array[1], dt)

            info_img=cv2.putText(info_img,"Camera Angles:",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,2)
            info_img=cv2.putText(info_img,str("{0:.1f}".format(180+kf.angle_x)),(10,180),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,2)
            info_img=cv2.putText(info_img,", "+str("{0:.1f}".format(kf.angle_y)),(90,180),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,2)
        
            # Creating PCD and Downsample ---- 10ms
            pcd = pc_handler.pcd_from_rgbd_cuda(color_image, depth_image)
            pcd = pc_handler.downsample_and_crop(pcd, voxel_size=down_sampling_voxel_size, minmax_Y_to_crop=2.5, tile_dim=tile_dim)
      
            # Record for plotting later
            profiler.record_frame(cycle_time_s=dt, num_points=len(pcd.point["positions"]))

            if pcd.point["positions"].shape[0] < 200:
                info_img=cv2.putText(info_img,"Frame Skipped:",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,2)
                info_img=cv2.putText(info_img,"No enough data",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,2)
                continue
            skip_this_frame=False

            # Rotate using matrices from your Kalman Filter
            R1, R2 = kf.get_rotation_matrices(dtype=pcd.point["positions"].dtype, device=pc_handler.device)
            pcd = pc_handler.rotate(pcd, R1, R2)

            # Translate and build map
            pcd, xyz_translate = pc_handler.translate_to_origin(pcd)

            start_timee = time.perf_counter()

            # Build the Height-Map ---- 1.5ms
            height_map_raw = pc_handler.build_height_map_gpu(pcd, tile_dim=tile_dim)

            hight_map=height_map_raw.clone()
            if (height_map_raw.shape[0]*height_map_raw.shape[1]) < 100 or not(                       # continue to next frame if hightmap is too small
                height_map_raw.shape[0]/5 < height_map_raw.shape[1] < height_map_raw.shape[0]*5):      # or if doesn't cover a good area
                skip_this_frame=True

            hight_map_np = hight_map.cpu().numpy().astype(np.float32)
            hight_map_np = cv2.normalize(hight_map_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            hight_map_np = cv2.resize(hight_map_np, (480, 480), interpolation=cv2.INTER_NEAREST)
            hight_map_np = cv2.cvtColor(hight_map_np, cv2.COLOR_GRAY2BGR)
            cv2.imshow("raw height map", hight_map_np)

            # Display the streams
            cv2.imshow("Color Frame", color_image)
            cv2.imshow("Depth Frame", depth_colormap)

            bbox_img=hight_map_np.copy()

            if skip_this_frame: #skip the frame after showing hightmap
                info_img=cv2.putText(info_img,"Frame Skipped",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,2)
                info_img=cv2.putText(info_img,"No enough height or points",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,2)
                continue
            
            # start_timee = time.perf_counter()
            # YOLO PREDICTION ---- 
            results = model.predict(hight_map_np, max_det=1, imgsz=hmap_img_size, conf=confidence, iou=0.7, verbose=False)
            
            all_corners = np.array([], np.float64)
            _ind = 0
            
            for r in results:
                for obb in r:
                    data = obb.obb.cpu().numpy()
                    corners = np.int32(np.reshape(data.xyxyxyxy, (-1, 1, 2)))

                    all_corners = np.append(all_corners, data.xyxyxyxy)
                    # _corners_txt = "corners xyxyxyxy:\n" + str(data.xyxyxyxy)
                    # _corners_txt2 = str(int(data.cls[0])) + " " + ' '.join(str(c) for c in data.xyxyxyxyn.flatten())

                    bbox_img = cv2.polylines(bbox_img, [corners], True, (0, 0, 255), 2)
                    bbox_img = cv2.putText(bbox_img, "Conf: " + str(np.int8(data.conf[0]*100)), (15, 15+_ind*45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, 2)
                    bbox_img = cv2.putText(bbox_img, "Class: " + obb.names[0], (15, 32+_ind*45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, 2)
                    _ind += 1

            if _ind < 1:
                info_img = cv2.putText(info_img, "Stairs Not Found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, 2)
            else:
                cv2.imshow("bounding box", bbox_img)

            # finishing_time = time.perf_counter()
            # inference_ms = (finishing_time - start_timee) * 1000
            # print(f"Heightmap build up time: {inference_ms:.2f} ms")

            # Roll back entire results to the main frame coordinates
            all_corners=np.reshape(all_corners, (-1, 4, 2))
            all_corners[:,:,0]=all_corners[:,:,0]*np.float64(height_map_raw.shape[1])*tile_dim/hmap_img_size
            all_corners[:,:,1]=all_corners[:,:,1]*np.float64(height_map_raw.shape[0])*tile_dim/hmap_img_size

            _found=False
            for corners in all_corners:
                stairs_cloud = pc_handler.crop_with_open3d_builtin(corners, pcd)
            
                if len(stairs_cloud.point["positions"]) < min_points_in_obb_area:
                    continue
                    
                # Filter points that are walls based on normal vectors
                radius = down_sampling_voxel_size * 3
                stairs_cloud, num_valid_points = pc_handler.filter_by_normal_angle(stairs_cloud, radius=radius)
                
                if num_valid_points < min_points_in_obb_area:
                    continue

                # live_viewer.update(stairs_cloud)

                dtype = stairs_cloud.point["positions"].dtype
                stairs_cloud.paint_uniform_color(o3d.core.Tensor([1.0, 1.0, 1.0], dtype=dtype, device=pc_handler.device))

                labels_tensor = stairs_cloud.cluster_dbscan(eps=0.080, min_points=10)
                p_labels = labels_tensor.cpu().numpy()


                # Extract individual step point clouds
                step_threshold = min_points_in_obb_area / 4
                stairs_points = stair_vis.extract_and_transform_steps(
                    stairs_cloud=stairs_cloud,
                    p_labels=p_labels,
                    min_points_threshold=step_threshold,
                    xyz_translate=xyz_translate,
                    kf=kf
                )

                # Default fallbacks in case stairs are not found
                overlay_img = color_image.copy()
                step_count = 0

                if len(stairs_points) >= 2:
                    # Project contours onto color frame
                    overlay_img, step_count = stair_vis.project_and_overlay(color_image, stairs_points)
                    cv2.imshow("stairs_contours", overlay_img)

                    # Update info window with step count and success status
                    info_img = stair_vis.draw_info(
                        info_img, dt, kf, num_steps=step_count, 
                        status_text="Stairs Found!", color=(0, 255, 0)
                    )
                else:
                    cv2.imshow("stairs_contours", color_image)
                    info_img = stair_vis.draw_info(
                        info_img, dt, kf, num_steps=0, 
                        status_text="Insufficient Steps", color=(0, 0, 255)
                    )


            # Check for exit or save keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 27 is the ESC key
                break
            elif key == ord('v'):
                stair_vis.save_frame_data(
                    color_image=color_image,
                    depth_colormap=depth_colormap,
                    height_map=hight_map_np,
                    bbox_img=bbox_img,
                    overlay_img=overlay_img,
                    info_img=info_img,
                    step_count=step_count,
                    kf=kf,
                    dt=dt
                )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Stopping camera and closing windows...")
        camera.stop()
        cv2.destroyAllWindows()
        profiler.plot_results()

if __name__ == "__main__":
    main()