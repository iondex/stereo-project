import cv2
import os
import glob
import numpy as np
import os.path as op
import sys
from camera_util import FindChessboardCorners, Calibrate

def Undistort(img):
    cameraMat, distCoeffs, _, _ = Calibrate(img)
    undistorted = cv2.undistort(img, cameraMat, distCoeffs)
    return undistorted

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["left", "right"]:
        subdir = "left"
    else:
        subdir = sys.argv[1]
        
    pics = glob.glob("../data/chessboard/%s/*" % subdir)
    for pic in pics:
        img = cv2.imread(pic)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        undistorted = Undistort(img)

        basename = op.basename(pic)
        cv2.imwrite(os.path.join("../result/undistorted/%s" % subdir, basename), undistorted)
        print(basename, "completed.")

if __name__ == "__main__":
    main()