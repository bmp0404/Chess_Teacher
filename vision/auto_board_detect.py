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


def reorder_corners(corners, nline, ncol):
    """ Reorders the chessboard corners by first grouping by y-coordinate, then sorting by x within each group (row). """
    # Reshape to (49, 2) for easier manipulation
    reshaped_arr = corners.reshape(-1, 2)

    # Sort by y-coordinate (to group into rows)
    sorted_by_y = reshaped_arr[np.argsort(reshaped_arr[:, 1])]

    # Now we need to group into rows and sort within each row by x-coordinate
    sorted_rows = []
    # Split into `nline` rows
    row_height = nline  # The number of points in a row (based on nline)
    for i in range(nline):
        # Extract one row at a time
        row = sorted_by_y[i * ncol:(i + 1) * ncol]
        # Sort by x-coordinate within the row
        row_sorted_by_x = row[np.argsort(row[:, 0])]
        sorted_rows.append(row_sorted_by_x)
    
    # Stack rows back into one array
    sorted_arr = np.vstack(sorted_rows)

    # Reshape back to (49, 1, 2)
    return sorted_arr.reshape(nline * ncol, 1, 2)





while True:
    # Capture frame-by-frame from the video stream
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    alpha = 3  # Contrast control (1.0-3.0)
    beta = 0.5  # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Convert the frame to grayscale
    gauss = cv2.GaussianBlur(adjusted, (7, 7), 0)
    gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

    # If we are not tracking, try to detect chessboard corners
    if not tracking_mode:
        ret, corners = cv2.findChessboardCorners(gray, (nline, ncol), None)

        # If corners are found, refine and draw them
        if ret:
            # Refine the corner positions to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw the detected corners
            cv2.drawChessboardCorners(frame, (nline, ncol), corners2, ret)

            # Store the detected corners and the current frame as the previous one for tracking
            prev_corners = corners2
            prev_gray = gray
            tracking_mode = True  # Enable tracking mode
        else:
            tracking_mode = False  # Disable tracking if detection fails and no corners were found

    else:
        # Track the corners using optical flow if chessboard corners were previously detected
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_corners, None, **lk_params)
        # print(new_corners.shape)
        new_corners = reorder_corners(new_corners, 7, 7)
        # If the corners were successfully tracked, update and draw them
        if new_corners is not None and status.sum() == len(prev_corners):
            prev_corners = new_corners  # Update the tracked corners
            prev_gray = gray  # Update the previous frame for optical flow

            # Draw tracked corners
            for i, corner in enumerate(new_corners):
                # print(i)
                x, y = corner.ravel()
                cv2.circle(frame, (int(x), int(y)), 5, (155*((i+1)/48), 155*((i+1)/48) +100, 155*((i+1)/48)), -1)
                if i == 0:
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
                if i == 48:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    


            # Extrapolate additional corners
            grid = new_corners.reshape((nline, ncol, 2))


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
