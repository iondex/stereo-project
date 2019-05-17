import cv2
import os
import glob
import numpy as np
import os.path as op

FILE = "data/left/left01.jpg"

def calibrate(img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    patternSize = (9, 6)
    retval, corners = cv2.findChessboardCorners(img, patternSize, None)
    if not retval:
        retval, corners = cv2.findChessboardCorners(img, patternSize, None)
    if not retval:
        raise Exception("Cannot find chessboard corners")

    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1,-1), criteria)

    object_points = []
    object_points = np.zeros((patternSize[1]*patternSize[0], 3), np.float32)
    object_points[:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1,2)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([object_points], [corners], img.shape[::-1], None, None)
    undistorted = cv2.undistort(img, cameraMatrix, distCoeffs)
    return retval, cameraMatrix, distCoeffs, rvecs, tvecs, undistorted

def main():
    pics = glob.glob("../data/left/*")
    for pic in pics:
        img = cv2.imread(pic)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, _, _, _, _, undistorted = calibrate(img)

        basename = op.basename(pic)
        cv2.imwrite(os.path.join("../result/basics/left", basename), undistorted)
        print(basename, "completed.")

if __name__ == "__main__":
    main()