import warnings
warnings.filterwarnings("ignore")

import pickle
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

# =========================
# LOAD MODEL
# =========================

with open('pushup.pkl', 'rb') as f:
    model = pickle.load(f)

# =========================
# LANDMARK NAMES
# =========================

landmarks = []

for val in range(1, 33 + 1):

    landmarks += [
        f'x{val}',
        f'y{val}',
        f'v{val}'
    ]

# =========================
# MEDIAPIPE SETUP
# =========================

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

# =========================
# REP VARIABLES
# =========================

counter = 0
stage = None
threshold = 0.80

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

        try:

            # CHECK PERSON
            if results.pose_landmarks is None:

                cv2.putText(
                    image,
                    "NO PERSON DETECTED",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow("Pushup Counter", image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                continue

            # EXTRACT LANDMARKS
            row = np.array(
                [[res.x, res.y, res.visibility]
                 for res in results.pose_landmarks.landmark]
            ).flatten()

            X = pd.DataFrame([row], columns=landmarks)

            # PREDICTION
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]

            probability = np.max(body_language_prob)

            # REP LOGIC

            if probability > threshold:

                # DOWN
                if body_language_class == 0:
                    stage = "down"

                # COUNT REP
                if body_language_class == 1 and stage == "down":

                    stage = "up"
                    counter += 1

            # UI BOX
            cv2.rectangle(
                image,
                (0, 0),
                (320, 110),
                (30, 30, 30),
                -1
            )

            # REPS
            cv2.putText(
                image,
                f'REPS: {counter}',
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # STAGE
            cv2.putText(
                image,
                f'STAGE: {stage}',
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # CONFIDENCE
            cv2.putText(
                image,
                f'CONF: {round(probability, 2)}',
                (180, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        except Exception as e:
            print("Error:", e)

        cv2.imshow("Pushup Counter", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()