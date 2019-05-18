import cv2
import os
import glob
import numpy as np
import os.path as op
import sys

def FindChessboardCorners(img):
    patternSize1 = (9, 6)
    patternSize2 = (6, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retval, corners = cv2.findChessboardCorners(img, patternSize1, None)
    patternSize = patternSize1
    if not retval:
        retval, corners = cv2.findChessboardCorners(img, patternSize2, None)
        patternSize = patternSize2

    if not retval:
        raise Exception("Cannot find chessboard corners.")

    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1,-1), criteria)
    return corners, patternSize

def Calibrate(img):
    corners, patternSize = FindChessboardCorners(img)

    object_points = []
    object_points = np.zeros((patternSize[1]*patternSize[0], 3), np.float32)
    object_points[:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1,2)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([object_points], [corners], img.shape[::-1], None, None)
    if not retval:
        raise Exception("calibrateCamera failed.")
    
    return cameraMatrix, distCoeffs, rvecs, tvecs

def Undistort(img):
    cameraMat, distCoeffs, _, _ = Calibrate(img)
    undistorted = cv2.undistort(img, cameraMat, distCoeffs)
    return undistorted

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["left", "right"]:
        subdir = "left"
    else:
        subdir = sys.argv[1]
        
    pics = glob.glob("../../data/%s/*" % subdir)
    for pic in pics:
        img = cv2.imread(pic)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        undistorted = Undistort(img)

        basename = op.basename(pic)
        cv2.imwrite(os.path.join("../../result/undistorted/%s" % subdir, basename), undistorted)
        print(basename, "completed.")

if __name__ == "__main__":
    main()