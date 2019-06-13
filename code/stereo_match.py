import numpy as np
import cv2 as cv

def main():
    print('Loading images...')
    imgL = cv.pyrDown(cv.imread("../data/aloe/aloeL.jpg")) 
    imgR = cv.pyrDown(cv.imread("../data/aloe/aloeR.jpg"))

    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('Computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv.imwrite("../result/disparity/aloe.jpg", disp)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')

if __name__ == '__main__':
    main()
    cv.destroyAllWindows()