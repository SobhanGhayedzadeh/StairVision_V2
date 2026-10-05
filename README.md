stair_detection_v2/
│
├── main.py                  # The single entry point. Parses CLI flags and orchestrates the loop.
├── requirements.txt         # Project dependencies.
│
├── configs/
│   ├── default.yaml         # Base parameters (tile_dim, down_sampling, hmap_img_size).
│   ├── edge_jetson.yaml     # Overrides for deployment (e.g., lower resolution, specific CUDA device).
│   └── test_offline.yaml    # Overrides for processing .bag files.
│
├── core/
│   ├── __init__.py
│   ├── camera.py            # RealSense pipeline handler. Switches seamlessly between live stream and .bag playback.
│   ├── imu_tracker.py       # Kalman filter class and IMU angle math.
│   ├── pointcloud.py        # Open3D CUDA functions, height map builders, and DBSCAN clustering.
│   └── yolo_detector.py     # YOLOv5-OBB model loading, inference, and GPU/CPU device management.
│
├── geometry/
│   ├── __init__.py
│   ├── math_utils.py        # Numba-jitted functions, vector math, PCA run direction, and OBB boundary checks.
│   └── stair_params.py      # Z-statistics, slope calculations, and tread surface gap logic.
│
└── utils/
    ├── __init__.py
    ├── visualization.py     # OpenCV drawing functions, bounding boxes, and color maps.
    ├── io.py                # File saving logic (PCDs, bounding box text files, dataset generation).
    └── platform_utils.py    # OS-level checks (e.g., Windows screen lock prevention).