import cv2
import numpy as np
import glob

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

def FindChessboardCornersList(imgList):
    imgPointsList1 = []
    imgPointsList2 = []
    objPointsList = []

    for img1, img2 in imgList:
        corners1, patternSize1 = FindChessboardCorners(img1)
        corners2, patternSize2 = FindChessboardCorners(img2)
        if patternSize1 != patternSize2:
            raise Exception("Invalid pair: patternSize not match.")
        
        imgPointsList1.append(corners1)
        imgPointsList2.append(corners2)

        objectPoints = [] # object points for this pair
        objectPoints = np.zeros((patternSize1[1]*patternSize1[0], 3), np.float32)
        objectPoints[:,:2] = np.mgrid[0:patternSize1[0], 0:patternSize1[1]].T.reshape(-1,2)
        objPointsList.append(objectPoints)

    return imgPointsList1, imgPointsList2, objPointsList

def Calibrate(img):
    corners, patternSize = FindChessboardCorners(img)

    object_points = []
    object_points = np.zeros((patternSize[1]*patternSize[0], 3), np.float32)
    object_points[:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1,2)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([object_points], [corners], img.shape[::-1], None, None)
    if not retval:
        raise Exception("calibrateCamera failed.")
    
    return cameraMatrix, distCoeffs, rvecs, tvecs

def StereoCalibrate(imgList):
    imgPointsList1, imgPointsList2, objPointsList = FindChessboardCornersList(imgList)
    imsize = (640, 480)

    retval, cameraMatrix1, distCoeffs1, _, _ = cv2.calibrateCamera(np.asarray(objPointsList), np.asarray(imgPointsList1), (640, 480), None, None)
    retval, cameraMatrix2, distCoeffs2, _, _ = cv2.calibrateCamera(np.asarray(objPointsList), np.asarray(imgPointsList2), (640, 480), None, None)

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, _, _, _, _ = \
        cv2.stereoCalibrate(objPointsList, imgPointsList1, imgPointsList2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imsize, flags=cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST)
    print(cameraMatrix1)

def main():
    def loadImage(fileName):
        orig = cv2.imread(fileName)
        im = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        return im

    imgList1 = [loadImage(fileName) for fileName in glob.glob("../../data/left/*")]
    imgList2 = [loadImage(fileName) for fileName in glob.glob("../../data/right/*")]
    imgList = zip(imgList1, imgList2)
    StereoCalibrate(imgList)

if __name__ == "__main__":
    main()