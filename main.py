import cv2
import mediapipe as mp
import math

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open the web camera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(frame_rgb)

    # Calculate and display angles
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get landmarks for left elbow angle
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        
        # Get landmarks for left hip angle
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        
        # Get landmarks for left knee extension angle
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # Calculate the angles
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        knee_extension_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Display the angles
        cv2.putText(frame, f'Elbow Angle: {int(elbow_angle)}', 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Hip Angle: {int(hip_angle)}', 
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Knee Extension Angle: {int(knee_extension_angle)}', 
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) 
            )

    # Display the frame
    cv2.imshow('Cycling Pose Estimation (Left)', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
