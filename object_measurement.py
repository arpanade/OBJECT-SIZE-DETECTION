import cv2
import numpy as np

def detect_object(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to segment the object
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise)
    min_contour_area = 100
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Continue only if a contour is found
    if contours:
        # Find the largest contour (outer boundary)
        largest_contour = max(contours, key=cv2.contourArea, default=None)

        # Calculate the bounding box for the largest contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Draw the bounding box on the frame
        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

        # Display the coordinates of the corners
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, point in enumerate(box):
            cv2.putText(frame, f"({point[0]}, {point[1]})", tuple(point), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        # Calculate the rotation angle
        angle = rect[2]

        # Display the rotation angle
        cv2.putText(frame, f"Rotation Angle: {angle:.2f} degrees", (10, 30), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return frame

if __name__ == "__main__":
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect the object and display results
        result_frame = detect_object(frame)

        # Display the frame
        cv2.imshow("Object Detection", result_frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
