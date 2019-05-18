import cv2
import numpy as np
import glob
import os.path as op

def FindChessboardCorners(img):
    patternSize = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retval, corners = cv2.findChessboardCorners(img, patternSize, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not retval:
        raise Exception("Cannot find chessboard corners.")

    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    return corners, patternSize

def FindChessboardCornersList(imgList):
    imgPointsList1 = []
    imgPointsList2 = []
    objPointsList = []

    for ((img1, _), (img2, _)) in imgList:
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
    print("Begin calibrating...")

    calib_flags = \
        cv2.CALIB_FIX_ASPECT_RATIO + \
        cv2.CALIB_ZERO_TANGENT_DIST + \
        cv2.CALIB_USE_INTRINSIC_GUESS + \
        cv2.CALIB_SAME_FOCAL_LENGTH + \
        cv2.CALIB_RATIONAL_MODEL + \
		cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5

    imgPointsList1, imgPointsList2, objPointsList = FindChessboardCornersList(imgList)
    imageSize = (640, 480)

    # retval, cameraMatrix1, distCoeffs1, _, _ = cv2.calibrateCamera(np.asarray(objPointsList), np.asarray(imgPointsList1), (640, 480), None, None)
    # retval, cameraMatrix2, distCoeffs2, _, _ = cv2.calibrateCamera(np.asarray(objPointsList), np.asarray(imgPointsList2), (640, 480), None, None)
    cameraMatrix1 = cv2.initCameraMatrix2D(np.asarray(objPointsList), np.asarray(imgPointsList1), imageSize, 0)
    cameraMatrix2 = cv2.initCameraMatrix2D(np.asarray(objPointsList), np.asarray(imgPointsList2), imageSize, 0)

    rmserror, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate(objPointsList, imgPointsList1, imgPointsList2, 
        cameraMatrix1, None, cameraMatrix2, None, imageSize, 
        flags=calib_flags)

    print("Result")
    print("RMS Error: ", rmserror)
    print("CameraMatrix1: ", cameraMatrix1)
    print("CameraMatrix2: ", cameraMatrix2)
    print("distCoeffs1: ", distCoeffs1)
    print("distCoeffs2: ", distCoeffs2)
    print("R: ", R)
    print("T: ", T)
    print("E: ", E)
    print("F: ", F)

    print("Begin rectification...")
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY)

    mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2)
    imgListL, imgListR = tuple(zip(*imgList))
    for (img, fileName) in imgListL:
        rectL = cv2.remap(img, mapL1, mapL2, cv2.INTER_LINEAR)
        cv2.imwrite("../../result/rectified/left/" + fileName, rectL)

    for (img, fileName) in imgListR:
        rectR = cv2.remap(img, mapR1, mapR2, cv2.INTER_LINEAR)
        cv2.imwrite("../../result/rectified/right/" + fileName, rectR)
    print("Rectification complete.")

def main():
    def loadImage(fileName):
        orig = cv2.imread(fileName)
        im = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        return im, op.basename(fileName)

    fileList1 = glob.glob("../../data/left/*")
    fileList2 = glob.glob("../../data/right/*")
    fileList1 = fileList1#[:len(fileList1) // 2]
    fileList2 = fileList2#[:len(fileList2) // 2]
    print("Begin reading images: ", len(fileList1))
    imgList1 = [loadImage(fileName) for fileName in fileList1]
    imgList2 = [loadImage(fileName) for fileName in fileList2]
    imgList = list(zip(imgList1, imgList2))
    StereoCalibrate(imgList)

if __name__ == "__main__":
    main()