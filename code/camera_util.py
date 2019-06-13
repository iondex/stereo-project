import cv2
import numpy as np
import glob
import os.path as op

CHESSBOARDIMAGES_DIR = "../data/chessboard/"

def FindChessboardCorners(img):
    patternSize = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retval, corners = cv2.findChessboardCorners(img, patternSize, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not retval:
        raise Exception("Cannot find chessboard corners.")

    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
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

def LoadChessboardImages():
    def loadImage(fileName):
        orig = cv2.imread(fileName)
        im = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        return im, op.basename(fileName)

    fileList1 = glob.glob(CHESSBOARDIMAGES_DIR + "left/*")
    fileList2 = glob.glob(CHESSBOARDIMAGES_DIR + "right/*")
    print("Begin reading images: ", len(fileList1))
    imgList1 = [loadImage(fileName) for fileName in fileList1]
    imgList2 = [loadImage(fileName) for fileName in fileList2]
    imgList = list(zip(imgList1, imgList2))
    return imgList

# def FindChessboardCorners(img):
#     patternSize = (9, 6)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#     retval, corners = cv2.findChessboardCorners(img, patternSize, None)
#     if not retval:
#         raise Exception("Cannot find chessboard corners.")

#     corners = cv2.cornerSubPix(img, corners, (11, 11), (-1,-1), criteria)
#     return corners, patternSize

# def Calibrate(img):
#     corners, patternSize = FindChessboardCorners(img)

#     objectPoints = []
#     objectPoints = np.zeros((patternSize[1]*patternSize[0], 3), np.float32)
#     objectPoints[:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1,2)

#     retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([objectPoints], [corners], img.shape[::-1], None, None)
#     if not retval:
#         raise Exception("calibrateCamera failed.")
    
#     return cameraMatrix, distCoeffs, rvecs, tvecs