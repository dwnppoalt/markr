import cv2
import cv2.aruco as aruco
import numpy as np

def calibrate_from_frame(frame, real_world_coords=None):
    if real_world_coords is None:
        # Using consistent coordinate system
        real_world_coords = {
            0: [0, 0],       # Top-left
            1: [300, 0],     # Top-right
            2: [300, 200],   # Bottom-right
            3: [0, 200],     # Bottom-left
        }
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        ids = ids.flatten()
        wb_corners = []
        wb_coords = []
        
        print(f"Detected markers: {ids}")
        print("Expected markers:", list(real_world_coords.keys()))
        
        detected_markers = {id: i for i, id in enumerate(ids)}
        print("Marker mapping:", detected_markers)
        
        for marker_id in real_world_coords:
            if marker_id in detected_markers:
                idx = detected_markers[marker_id]
                c = corners[idx][0].mean(axis=0)
                wb_corners.append(c)
                wb_coords.append(real_world_coords[marker_id])
                print(f"Added marker {marker_id} at position {c}")
            else:
                print(f"Missing marker {marker_id}")
        
        print(f"Collected {len(wb_corners)} corners out of 4 needed")
        
        if len(wb_corners) == 4:
            print("Found all 4 corners, calculating homography...")
            sorted_pairs = sorted(zip(wb_coords, wb_corners), key=lambda x: x[0])
            src_pts = np.array([p[1] for p in sorted_pairs], dtype='float32')
            dst_pts = np.array([p[0] for p in sorted_pairs], dtype='float32')
            
            H, status = cv2.findHomography(src_pts, dst_pts)
            if status is not None:
                for pt in src_pts:
                    cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
                
                warped_size = (int(max(dst_pts[:,0])), int(max(dst_pts[:,1])))
                warped = cv2.warpPerspective(frame, H, warped_size)
                print(f"Successfully warped image to size {warped_size}")
                return warped
            else:
                print("Failed to calculate homography")
        else:
            print(f"Not enough corners: found {len(wb_corners)}/4")
            print("Missing markers:", [id for id in real_world_coords if id not in detected_markers])
    else:
        print("No markers detected in calibrate_from_frame")
    return None
