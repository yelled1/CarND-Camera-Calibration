import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
#objpoints = dist_pickle["objpoints"]
#imgpoints = dist_pickle["imgpoints"]
objpoints = dist_pickle["dist"]
imgpoints = dist_pickle["mtx"]

# Read in an image
img = cv2.imread('test_image.png')

# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image

def cal_undistort(img, objpoints, imgpoints, nb_x, nb_y):
    #Finding chessboard corners (for an 8x6 board):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nb_x, nb_y), None)
    #Drawing detected corners on an image:
    img = cv2.drawChessboardCorners(img, (nb_x, nb_y, corners, ret))
    #Camera calibration:
    #given object points, image points, & the shape of the grayscale image:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Use cv2.calibrateCamera and cv2.undistort()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

"""
imgpoints = []
objp = np.zeros(( nb_x * nb_y, 3), np.float32)
print(objp.shape)
#objpoints
objp[:,:2] = np.mgrid[:nb_x, :nb_y].T.reshape(-1, 2)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nb_x, nb_y), None)
if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)
"""
undistorted = cal_undistort(img, objpoints, imgpoints, nb_x=8, nb_y=6)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
