import numpy as np
import open3d as o3d

class KalmanFilter:
    def __init__(self, Q_angle=0.001, Q_bias=0.003, R_measure=0.03):
        # Initialize Kalman filter variables
        self.Q_angle = Q_angle
        self.Q_bias = Q_bias
        self.R_measure = R_measure
        
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.bias_x = 0.0
        self.bias_y = 0.0
        
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # State covariance matrix

    def update(self, accel_angle_x, accel_angle_y, gyro_x, gyro_y, dt):
        # Update angle_x
        rate_x = gyro_x - self.bias_x
        self.angle_x += dt * rate_x

        # Prediction step
        self.P += np.array([[self.Q_angle, 0.0], [0.0, self.Q_bias]], dtype=np.float32)

        # Measurement update for angle_x
        self._measurement_update(accel_angle_x, gyro_x)

        # Update angle_y
        rate_y = gyro_y - self.bias_y
        self.angle_y += dt * rate_y

        # Measurement update for angle_y
        self._measurement_update(accel_angle_y, gyro_y, is_angle_y=True)

    def _measurement_update(self, angle, gyro, is_angle_y=False):
        """Perform the measurement update step for the Kalman filter."""
        # Determine which angle to update
        if is_angle_y:
            angle_est = self.angle_y
            bias = self.bias_y
        else:
            angle_est = self.angle_x
            bias = self.bias_x

        # Calculate the Kalman gain and update the angle and bias
        y = angle - angle_est
        K = self.P[0, 0] / (self.P[0, 0] + self.R_measure)
        angle_est += K * y
        bias += K * (gyro - bias)
        self.P[0, 0] -= K * self.P[0, 0]

        # Assign updated values back
        if is_angle_y:
            self.angle_y = angle_est
            self.bias_y = bias
        else:
            self.angle_x = angle_est
            self.bias_x = bias

    def get_normal_vector(self):
        """Calculate the adjusted normal vector based on the angles."""
        normal_vector = np.array([
            np.sin(np.radians(self.angle_y)),  # X component
            np.sin(np.radians(self.angle_x)),  # Y component
            np.cos(np.radians(self.angle_x)) * np.cos(np.radians(self.angle_y))  # Z component
        ], dtype=np.float32)

        # Normalize the normal vector
        norm = np.linalg.norm(normal_vector)
        return normal_vector / norm if norm != 0 else normal_vector
    
    def get_rotation_matrices(self, dtype, device):
        """
        Calculates and returns the Open3D rotation matrices R1 and R2 
        based on the current Kalman filter angles.
        """
        # Calculate angles in radians[cite: 1]
        R_angles = np.array([180.0 + self.angle_x, self.angle_y])
        R_angles = np.deg2rad(R_angles)

        c1, s1 = np.cos(R_angles[1]), np.sin(R_angles[1])
        c2, s2 = np.cos(R_angles[0]), np.sin(R_angles[0])

        # Create Open3D tensors[cite: 1]
        R1 = o3d.core.Tensor(
            [[ c1, -s1, 0.0],
             [ s1,  c1, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device
        )

        R2 = o3d.core.Tensor(
            [[1.0, 0.0, 0.0],
             [0.0, c2, -s2],
             [0.0, s2, c2]],
            dtype=dtype,
            device=device
        )

        return R1, R2
    
    def get_inverse_rotation_matrices(self, dtype, device):
        """Calculates reverse rotation matrices to return points to camera frame."""
        R_angles = np.deg2rad(np.array([180.0 + self.angle_x, self.angle_y]))

        c1, s1 = np.cos(-R_angles[1]), np.sin(-R_angles[1])
        c2, s2 = np.cos(-R_angles[0]), np.sin(-R_angles[0])

        R1_inv = o3d.core.Tensor(
            [[ c1, -s1, 0.0],
             [ s1,  c1, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device
        )

        R2_inv = o3d.core.Tensor(
            [[1.0, 0.0, 0.0],
             [0.0, c2, -s2],
             [0.0, s2, c2]],
            dtype=dtype,
            device=device
        )

        return R1_inv, R2_inv
