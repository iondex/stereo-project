# stereo-project
Stereo project implementation using Python and OpenCV-Python.

## Directory structure
* `/code/`: Implementation codes written in python.
  * `/code/monocular.py`: Part one -- Basic camera calibration and undistortion.
  * `/code/stereo_calib.py`: Part two -- Stereo calibration and rectification. (WIP)
  * `/code/stereo_match.py`: Part three -- (WIP)
* `/data`: Original data in the two zip file.
* `/result`: Program result folder.
  * `/result/undistorted`: Undistorted result from Part-One
  * `/result/rectified`: Rectification result from Part-Two

## How to use -- Calibration and Distortion
```shell
git clone http://github.com/iondex/stereo-project
cd stereo-project/code/
python calib.py left
# results are in results/undistorted/left
```

## How to use -- Stereo Calibrate and Rectification
```shell
cd stereo-project/code/
python stereo_calib.py
# results are in results/rectified
```