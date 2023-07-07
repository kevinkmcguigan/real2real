# real2real

`real2real` is a repository containing sample code to match a photo to a sparse LiDAR point cloud. The goal is to estimate the photo's location, orientation, and lens distortion parameters using two different approaches: linear and spherical projections.

## Files

1. `match_photo2lidar_linear.py` - This script uses a linear approach to match the photo's keypoints to the LiDAR point cloud. It creates 3D lines in the Cartesian coordinate system by projecting the keypoints from the photo's focal point and then matches the LiDAR points to the nearest 3D lines. The camera parameters are optimized to minimize the sum of the squared distances between the LiDAR points and the corresponding 3D lines.

2. `match_photo2lidar_spherical.py` - This script uses a spherical approach to match the photo's keypoints to the LiDAR point cloud. It projects the LiDAR points and 2D keypoints to spherical coordinates, and then matches the keypoints to the LiDAR points. The camera parameters are optimized to minimize the difference between the keypoints and the corresponding LiDAR points in the spherical coordinate system.

3. 'match_point2point_ICP.py' - This script employs the Iterative Closest Point (ICP) algorithm to align two sets of LiDAR point clouds, termed A and B. The ICP algorithm works by iteratively revising the transformation needed to minimize the distance between the points in A and their closest counterparts in B. This script outputs the final transformation matrix that best aligns point cloud A to B, along with the Euclidean distances (errors) of the nearest neighbor for each point in the now-aligned A. Note that this basic implementation of ICP doesn't handle outliers and assumes a one-to-one correspondence between points in A and B, which might not always be the case.

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

