'''Rough outline ofmethod to match a single photo of known location to lidar point cloud by sperical projection'''

import cv2
import numpy as np
from scipy.optimize import least_squares

# Load the LiDAR point cloud (assuming it is in a numpy array format)
lidar_points = np.load("lidar_points.npy")

# Load the photo
photo = cv2.imread("photo.jpg")

# Extract features from the photo using the ORB feature extractor
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(photo, None)

# Convert keypoints to numpy array
keypoints_2d = np.array([kp.pt for kp in keypoints])

def project_keypoints_to_spherical(keypoints_2d, camera_params):
    # Unpack camera parameters
    focal_length_x, focal_length_y, principal_point_x, principal_point_y, *extrinsic_params, *distortion_params = camera_params

    # Define the camera matrix
    K = np.array([
        [focal_length_x, 0, principal_point_x],
        [0, focal_length_y, principal_point_y],
        [0, 0, 1]
    ])

    # Define the distortion coefficients
    distortion_coefficients = np.array(distortion_params)

    # Undistort the keypoints
    undistorted_keypoints = cv2.undistortPoints(np.expand_dims(keypoints_2d, axis=1), K, distortion_coefficients)

    # Convert undistorted keypoints to normalized image coordinates
    normalized_keypoints = np.squeeze(undistorted_keypoints)

    # Convert the extrinsic parameters to rotation and translation vectors
    rotation_vector = np.array(extrinsic_params[:3])
    translation_vector = np.array(extrinsic_params[3:])

    # Compute the 3D points in the camera coordinate system
    z_values = np.ones((normalized_keypoints.shape[0], 1))
    keypoints_3d_camera = np.hstack([normalized_keypoints, z_values])

    # Compute the 3D points in the world coordinate system
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    keypoints_3d_world = np.dot(keypoints_3d_camera, rotation_matrix.T) + translation_vector

    # Convert the 3D keypoints to spherical coordinates
    r = np.sqrt(np.sum(keypoints_3d_world**2, axis=1))
    theta = np.arccos(keypoints_3d_world[:, 2] / r)
    phi = np.arctan2(keypoints_3d_world[:, 1], keypoints_3d_world[:, 0])

    keypoints_spherical = np.column_stack([r, theta, phi])

    return keypoints_spherical

# Define a function that computes the difference between LiDAR points and projected keypoints
def compute_residuals(params, lidar_points, keypoints_2d):
    # Update camera_params with the optimized parameters
    camera_params = update_camera_params(params)

    # Project the keypoints to spherical coordinates
    keypoints_spherical = project_keypoints_to_spherical(keypoints_2d, camera_params)

    # Calculate the residuals between the LiDAR points and the projected keypoints
    residuals = keypoints_spherical - lidar_points

    return residuals

# Initialize the camera parameters and lens distortion parameters
initial_camera_params = np.array([...])  # Intrinsic and extrinsic parameters
initial_lens_distortion_params = np.array([...])  # Lens distortion parameters
initial_params = np.concatenate((initial_camera_params, initial_lens_distortion_params))

# Optimize the camera parameters and lens distortion parameters using least squares
result = least_squares(compute_residuals, initial_params, args=(lidar_points, keypoints_2d))

# Extract the optimized camera parameters and lens distortion parameters
optimized_params = result.x
optimized_camera_params = optimized_params[:len(initial_camera_params)]
optimized_lens_distortion_params = optimized_params[len(initial_camera_params):]

print("Optimized Camera Parameters:", optimized_camera_params)
print("Optimized Lens Distortion Parameters:", optimized_lens_distortion_params)
