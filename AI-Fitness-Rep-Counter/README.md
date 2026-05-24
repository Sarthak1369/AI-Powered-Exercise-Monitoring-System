\# AI Fitness Rep Counter



A real-time AI-powered fitness exercise monitoring system using MediaPipe, OpenCV, and Machine Learning.



\## Features



\- Real-time pose detection

\- Pushup repetition counting

\- Pose landmark extraction

\- Machine learning classification

\- Random Forest based prediction

\- Live webcam feedback



\## Technologies Used



\- Python

\- OpenCV

\- MediaPipe

\- Scikit-learn

\- NumPy

\- Pandas



\## Project Workflow



1\. Capture exercise data

2\. Extract pose landmarks using MediaPipe

3\. Save landmarks into CSV dataset

4\. Train machine learning models

5\. Save trained Random Forest model

6\. Run real-time prediction and rep counting



\## Files



\### collect\_data.py



Used to:

\- Read pushup video

\- Detect body landmarks

\- Store landmark coordinates into CSV dataset



\### train\_model.py



Used to:

\- Train multiple ML models

\- Compare model performance

\- Save best model using Pickle



\### realtime\_counter.py



Used to:

\- Load trained ML model

\- Detect pose in real time

\- Predict pushup state

\- Count repetitions



\## Installation



Install required libraries:



pip install -r requirements.txt



\## Run Dataset Collection



python src/collect\_data.py



\## Train Model



python src/train\_model.py



\## Run Real-Time Counter



python src/realtime\_counter.py



\## Author



Sarthak Gupta

