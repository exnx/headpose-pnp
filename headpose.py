# import the necessary packages
from imutils import face_utils  # open souce tool, MIT License
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
import math


# model a generic face in 3D world coordinates
model_points = np.array([
                    (0.0, 0.0, 0.0),             # Nose tip
                    (0.0, -330.0, -65.0),        # Chin
                    (-225.0, 170.0, -135.0),     # Left eye left corner
                    (225.0, 170.0, -135.0),      # Right eye right corne
                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                    (150.0, -150.0, -125.0)      # Right mouth corner
                ])

# camera intrinsics constants
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion, these are 0's

# dlib landmark indices for 6 key points (nose,chin,lefteye,righteye,leftmount,rightmouth)
landmark_index = [30,8,36,45,48,54]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# start video stream from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()  # get the size of the image output (for camera intrinsics calc)

size = frame.shape


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def stream_video():
    # loop over frames from the video stream
    while True:
        # get frame from video file stream and convert it to grayscale)
        ret, frame = cap.read()

        handle_images(frame)


        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

def handle_images(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = detector(gray, 0)

    # if no faces found, not paying attention
    if not len(faces) > 0:
        cv2.putText(frame, "Way distracted!", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # loop over the face detections
    for (i, face) in enumerate(faces):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # extract 6 key landmarks around the face for head post estimation
        six_landmarks = []


        # convert each landmark to tuple
        for ind in landmark_index:
            six_landmarks.append(tuple(shape[ind].tolist()))

        # convert 6 landmarks to numpy array, these represent points in image plane
        image_points = np.array(six_landmarks, dtype="double")

        # Camera internals, estimated by the size of the photo frame (at beg.)
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )

        # print("Camera Matrix :\n {0}".format(camera_matrix))

        # solve for the rotation and translation vectors (sucess is boolean)
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rotation_matrix = np.zeros((3,3))

        rotation_matrix = cv2.Rodrigues(rotation_vector,rotation_matrix)

        euler_angles = rotationMatrixToEulerAngles(rotation_matrix[0])

        euler_angles = np.degrees(euler_angles)

        print(euler_angles)

        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))

        # print the 6 points we're tracking on the frame
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose to estimate gaze
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # p1 is the starting point, p2 is the point in front
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # draw the line
        cv2.line(frame, p1, p2, (255,0,0), 2)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # show the x and y rotation
        cv2.putText(frame, "X rotation: {}, Y rotation: {}, Z rotation: {}".format(euler_angles[0],euler_angles[1], euler_angles[2]), (x - 10, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # if angles too large, not paying attention
        if abs(euler_angles[1]) > 35 or (euler_angles[0] < 0 and euler_angles[0] > -175) or (euler_angles[0] > 0 and euler_angles[0] < 168):
            cv2.putText(frame, "Not paying attention!", (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Is paying attention!", (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (xdot, ydot) in six_landmarks:
            cv2.circle(frame, (xdot, ydot), 3, (0, 0, 255), -1)

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", frame)

# if __name__ == "__main__":
# headposer = HeadPose()

stream_video()




# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
