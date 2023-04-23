# real2real

`real2real` is a repository containing sample code to match a photo to a sparse LiDAR point cloud. The goal is to estimate the photo's location, orientation, and lens distortion parameters using two different approaches: linear and spherical projections.

## Files

1. `match_photo2lidar_linear.py` - This script uses a linear approach to match the photo's keypoints to the LiDAR point cloud. It creates 3D lines in the Cartesian coordinate system by projecting the keypoints from the photo's focal point and then matches the LiDAR points to the nearest 3D lines. The camera parameters are optimized to minimize the sum of the squared distances between the LiDAR points and the corresponding 3D lines.

2. `match_photo2lidar_spherical.py` - This script uses a spherical approach to match the photo's keypoints to the LiDAR point cloud. It projects the LiDAR points and 2D keypoints to spherical coordinates, and then matches the keypoints to the LiDAR points. The camera parameters are optimized to minimize the difference between the keypoints and the corresponding LiDAR points in the spherical coordinate system.

## Usage

To run the scripts, make sure you have the required dependencies installed (e.g., OpenCV, NumPy, and SciPy). Update the sample code with your own LiDAR point cloud data and photo. You may need to adjust the feature extraction method, coordinate transformation steps, and optimization settings to achieve the best results for your specific data.

```bash
python match_photo2lidar_linear.py
```

```bash
python match_photo2lidar_spherical.py
```

## Dependencies

- OpenCV
- NumPy
- SciPy

## License

MIT

