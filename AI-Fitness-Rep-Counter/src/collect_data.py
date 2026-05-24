import mediapipe as mp
import cv2
import numpy as np
import csv

# =========================
# CREATE CSV HEADER
# =========================

landmarks = ['class']

for val in range(1, 33 + 1):
    landmarks += [
        f'x{val}',
        f'y{val}',
        f'v{val}'
    ]

with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(
        f,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL
    )

    csv_writer.writerow(landmarks)

# =========================
# EXPORT LANDMARKS
# =========================

def export_landmarks(results, action):

    try:

        action_code = 1 if action == "up" else 0

        keypoints = np.array(
            [[res.x, res.y, res.visibility]
             for res in results.pose_landmarks.landmark]
        ).flatten()

        keypoints = np.insert(keypoints, 0, action_code)

        with open('coords.csv', mode='a', newline='') as f:

            csv_writer = csv.writer(
                f,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )

            csv_writer.writerow(keypoints)

        print(f"{action} saved")

    except Exception as e:
        print("Error:", e)

# =========================
# MEDIAPIPE SETUP
# =========================

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# =========================
# VIDEO INPUT
# =========================

cap = cv2.VideoCapture('pushups2.avi')

# =========================
# MAIN LOOP
# =========================

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # DRAW LANDMARKS
        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(245, 117, 66),
                    thickness=2,
                    circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230),
                    thickness=2,
                    circle_radius=2
                )
            )

        cv2.putText(
            image,
            "Press U = UP | D = DOWN",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        key = cv2.waitKey(10)

        # SAVE UP
        if key == ord('u'):
            export_landmarks(results, 'up')

        # SAVE DOWN
        if key == ord('d'):
            export_landmarks(results, 'down')

        cv2.imshow("Data Collection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()