import numpy as np
import cv2

nline = 7  # Number of internal corners along rows
ncol = 7  # Number of internal corners along columns

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize video capture (0 is the default camera, adjust if necessary)
video_path = './chessvid_flipped.mp4'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

prev_corners = None
prev_gray = None
tracking_mode = False

def distance (p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def leftmost_point(corners):
    """Finds the point with the lowest x-coordinate in a 7x7x2 numpy array."""
    min_x = float('inf')
    second_min_x = float('inf')
    leftmost = None
    second_leftmost = None
    for row in corners:
        for point in row:
            if point[0] < min_x:
                second_min_x = min_x
                second_leftmost = leftmost
                min_x = point[0]
                leftmost = point
            elif point[0] < second_min_x:
                second_min_x = point[0]
                second_leftmost = point
    return leftmost, second_leftmost

def delta_y(row):
    """Returns the difference between the maximum and minimum y-coordinates in a 7x2 numpy array."""
    y_coords = row[: -1]
    # print(y_coords)
    return np.max(y_coords) - np.min(y_coords)

def reorder_corners(corners, nline, ncol, middle):
    """ Reorders the chessboard corners by first grouping by y-coordinate, then sorting by x within each group (row). """
    # print(corners[0])
    distance_center = [(x, distance(x[0], middle[0])) for x in corners]
    distance_center.sort(reverse=True, key=lambda x: x[1])
    distance_center = [x[0] for x in distance_center]
    return np.array(distance_center)
    
    # Reshape to (49, 2) for easier manipulation
    # reshaped_arr = corners.reshape(-1, 2)
    
    # Sort by y-coordinate (to group into rows)
    # sorted_by_y = reshaped_arr[np.argsort(reshaped_arr[:, 1])]

    # Now we need to group into rows and sort within each row by x-coordinate
    # sorted_rows = []
    # Split into `nline` rows
    # row_height = nline  # The number of points in a row (based on nline)
    # for i in range(nline):
    #     # Extract one row at a time
    #     row = sorted_by_y[i * ncol:(i + 1) * ncol]
    #     # Sort by x-coordinate within the row
    #     row_sorted_by_x = row[np.argsort(row[:, 0])]
    #     sorted_rows.append(row_sorted_by_x)
    
    # # Stack rows back into one array
    # sorted_arr = np.vstack(sorted_rows)

    # # Reshape back to (49, 1, 2)
    # return sorted_arr.reshape(nline * ncol, 1, 2)




already_called = 1
while True:
    # Capture frame-by-frame from the video stream
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break


    # preprocessing 
    alpha = 3  # Contrast control (1.0-3.0)
    beta = 0.5  # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Convert the frame to grayscale
    gauss = cv2.GaussianBlur(adjusted, (7, 7), 0)
    gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
    two_d_corners = [[0]]
    # If we are not tracking, try to detect chessboard corners
    if not tracking_mode:
        ret, corners = cv2.findChessboardCorners(gray, (nline, ncol), None)
        # If corners are found, refine and draw them
        if ret:
            # print(ret)
            # Refine the corner positions to sub-pixel accuracy
            # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw the detected corners
            cv2.drawChessboardCorners(frame, (nline, ncol), corners, ret)

            # Store the detected corners and the current frame as the previous one for tracking
            prev_corners = corners
            prev_gray = gray
            already_called = 1
            tracking_mode = True  # Enable tracking mode
        else:
            tracking_mode = False  # Disable tracking if detection fails and no corners were found

    else:
        two_d_corners = prev_corners.reshape(-1, 2).reshape(7,7,2)

        
        # Track the corners using optical flow if chessboard corners were previously detected
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_corners, None, **lk_params)
        # new_corners = reorder_corners(new_corners, 7, 7, new_corners[24 * already_called])
    


        # 
        for i in range(len(two_d_corners)):
            for j in range(len(two_d_corners[i])):
                min_dist = -1
                min_point = None
                two_d_corners[i][j] = min(new_corners, key=lambda x: distance(two_d_corners[i][j], x[0]))
            
        leftmost, second_leftmost = leftmost_point(two_d_corners)
        while leftmost not in two_d_corners[0] or second_leftmost not in two_d_corners[0]:
            two_d_corners = np.rot90(two_d_corners)



        already_called = 2
        # If the corners were successfully tracked, update and draw them
        if new_corners is not None and status.sum() == len(prev_corners):
            prev_corners = new_corners  # Update the tracked corners
            prev_gray = gray  # Update the previous frame for optical flow

            colors = {
                0: (255, 0, 0),
                1: (0, 255, 0),
                2: (0, 0, 255),
                3: (255, 255, 0),
                4: (255, 0, 255),
                5: (0, 255, 255),
                6: (255, 255, 255)
            }

            for i in range(len(two_d_corners)):
                for j in range(len(two_d_corners[i])):
                    if np.array_equal(two_d_corners[i][j], leftmost) or np.array_equal(two_d_corners[i][j], second_leftmost):
                        # print(i)
                        cv2.circle(frame, (int(two_d_corners[i][j][0]), int(two_d_corners[i][j][1])), 5, (200,100,200))
                    else:
                        cv2.circle(frame, (int(two_d_corners[i][j][0]), int(two_d_corners[i][j][1])), 5, colors[i], -1)


            square_size = 50  # Set this to the actual size of a square on the chessboard in your desired output
            destination_corners = np.array([
                [1, 1],
                [square_size * (7 - 1), 1],
                [square_size * (7 - 1), square_size * (7- 1)],
                [1, square_size * (7 - 1)]
            ], dtype='float32')

            # print(two_d_corners.shape)
            # source_corners = np.array(two_d_corners, dtype="float32")
            print(corners.shape)
            origin_corners = two_d_corners.reshape((49,1,2))
            source_corners = np.array([
                origin_corners[0][0],         # Top-left corner
                origin_corners[7- 1][0],  # Top-right corner
                origin_corners[-1][0],        # Bottom-right corner
                origin_corners[-7][0]      # Bottom-left corner
            ], dtype='float32')
            
            # Find the homography matrix
            H, _ = cv2.findHomography(source_corners, destination_corners)
            # print(H)
            warped_image = cv2.warpPerspective(frame, H, (square_size * 8, square_size * 8))
            inv_h = np.linalg.inv(H)
            newPoints = []
            resizeFactor = -1/8    
            newPoints.append(np.array([warped_image.shape[0] * (1 + resizeFactor), warped_image.shape[1] * (1 + resizeFactor), 1], dtype=np.float32).reshape(-1, 1))
            newPoints.append(np.array([warped_image.shape[0] * (1 + resizeFactor), warped_image.shape[1] * (resizeFactor), 1], dtype=np.float32).reshape(-1, 1))
            newPoints.append(np.array([warped_image.shape[0] * (resizeFactor), warped_image.shape[1] * (resizeFactor), 1], dtype=np.float32).reshape(-1, 1))  
            newPoints.append(np.array([warped_image.shape[0] * (resizeFactor), warped_image.shape[1] * (1 + resizeFactor), 1], dtype=np.float32).reshape(-1, 1))      
            print(newPoints)
            res = []
            for point in newPoints:
                point = inv_h @ point
                x = point[0][0] / point[2][0]
                y = point[1][0] / point[2][0]
            # tempPoint = (cv2.perspectiveTransform(np.array([[[0,0]]]), inv_h))[0][0]
                res.append((int(x), int(y)))
            # warped_image = cv2.warpPerspective(frame, H, (square_size * 8, square_size * 8))
            cv2.imshow('Warped Chessboard', warped_image)
            print(newPoints[0][0],newPoints[0][1])
            print(f"Frame Width: {frame.shape[1]}, Frame Height: {frame.shape[0]}")
            for tempPoint in res:
                cv2.circle(frame, (tempPoint[0], tempPoint[1]), 5, (255,255,255))
            source_corners = np.array(res, dtype='float32')
            destination_corners = np.array([
                [0, 0],
                [square_size * (8), 0],
                [square_size * (8), square_size * (8)],
                [0, square_size * (8)]
            ], dtype='float32')

            H, _ = cv2.findHomography(source_corners, destination_corners)
            # print(H)
            warped_image = cv2.warpPerspective(frame, H, (square_size * 8, square_size * 8))
            cv2.imshow('Warped Chessboard', warped_image)
            # Draw tracked corners
            # for i, corner in enumerate(new_corners):
            #     # print(i)
            #     x, y = corner.ravel()
            #     cv2.circle(frame, (int(x), int(y)), 5, (255,255,255), -1)#(155*((i+1)/48), 155*((i+1)/48) +100, 155*((i+1)/48)), -1)
            #     if i == 0:
            #         cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
            #     # if i == 2:
            #     #     cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            #     # if i == 3:
            #     #     cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
            #     if i == 48:
            #         cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    


            # Extrapolate additional corners
            # grid = new_corners.reshape((nline, ncol, 2))


            # cv2.circle(frame, tuple(extrapolated_top_left.astype(int)), 8, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(extrapolated_top_right.astype(int)), 8, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(extrapolated_bottom_left.astype(int)), 8, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(extrapolated_bottom_right.astype(int)), 8, (0, 0, 255), -1)
        else:
            # If tracking fails, try detecting again
            tracking_mode = False

    # Display the resulting frame with corners
    cv2.imshow('Chessboard Corners Live', frame)

    

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
