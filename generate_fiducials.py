import os
import cv2
import cv2.aruco as aruco

os.makedirs("fiducials", exist_ok=True)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

for i in range(4):
    marker = aruco.generateImageMarker(aruco_dict, i, 200)
    cv2.imwrite(f"fiducials/fiducial_{i}.png", marker)
