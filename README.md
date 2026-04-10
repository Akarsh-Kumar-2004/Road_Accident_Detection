# Accident Detection System

**Last Updated: *7 June 2022***

1. Demonstration
2. What is Accident Detection System?
3. Prerequisites
4. Getting Started- How to use it?
5. Description
6. Future Work

## 1. Demonstration

![Demo](https://user-images.githubusercontent.com/54409969/173066273-732f7da9-8645-4809-aa7a-bb2f78548b3e.gif)

## 2. What is Accident Detection System?

An accident Detection System is designed to detect accidents via video or CCTV footage. Road accidents are a significant problem for the whole world. Many people lose their lives in road accidents. We can minimize this issue by using CCTV accident detection. This repository majorly explores how CCTV can detect these accidents with the help of Deep Learning.

## 3. Prerequisites

- To use this project Python Version > 3.6 is recommended.
- To contribute to this project, knowledge of basic python scripting, Machine Learning, and Deep Learning will help.

## 4. Getting Started - How to use it?

### Clone this repository

`https://github.com/krishrustagi/Accident-Detection-System.git`

To install all the packages required to run this python program
`pip install -r requirements.txt`

**Note:** This project requires a camera. So make sure you have a connected camera to your device. You can also use a downloaded video if not using a camera.

### Run
Before running the program, you need to run the `accident-classification.ipynb` file which will create the `model_weights.h5` file. Then, to run this python program, you need to execute the `main.py` python file.

## 5. Description

This program includes 4 things.

1. `data`: Kaggle dataset on [Accident Detection from CCTV footage](https://www.kaggle.com/code/mrcruise/accident-classification/data).
2. `accident-classification.ipynb`: This is a jupyter notebook that generates a model to classify the above data. This file generates two important files `model.json` and `model_weights.h5`.
3. `detection.py`: This file loads the Accident Detection system with the help of `model.json` and `model_weights.h5` files.
4. `camera.py`: It packs the camera and executes the `detection.py` file on the video dividing it frame by frame and displaying the percentage of the prediction in the accident (if present) in the frame.

## 6. Upgraded Pipeline

The original repository is frame-level classification only. The improved runtime in this workspace now adds:

1. `incident_pipeline.py`: YOLOv8-based vehicle detection plus accident-understanding heuristics for collision zones, fire/smoke, and possible overturned vehicles.
2. `upgraded_camera.py`: sequence-aware alerting that scores multiple frames over time and only raises an alert when the temporal signal is strong enough.
3. `location_services.py`: resolves the current PC location automatically by trying Windows location services first, then IP geolocation as a fallback, and finds nearby hospitals and route links.
4. `enhanced_map_app.py`: shows the incident point, nearest hospitals, and a route on an interactive map.

## 7. How To Run The Upgraded Version

Install dependencies:

`pip install -r requirements.txt`

Run the upgraded detector:

`python main.py`

Run the enhanced map:

`streamlit run enhanced_map_app.py`

Optional environment variables for email alerts:

`ACCIDENT_ALERT_EMAIL_SENDER`
`ACCIDENT_ALERT_EMAIL_PASSWORD`
`ACCIDENT_ALERT_EMAIL_RECEIVER`
`ACCIDENT_CAMERA_ID`
`ACCIDENT_TEMPORAL_WEIGHTS`

## 8. Notes

1. The temporal module supports a GRU path when you later provide trained weights through `ACCIDENT_TEMPORAL_WEIGHTS`.
2. Vehicle detection uses pretrained YOLOv8. Collision-zone, fire/smoke, and overturn understanding are implemented as practical visual heuristics until you train a dedicated accident dataset for those classes.
3. PC location is resolved automatically at runtime. On Windows, the app first tries the device's own location service. If Windows location access is disabled or unavailable, it falls back to IP-based geolocation, and if that also fails it uses the configured default coordinates.
4. If you want the most accurate "from my PC" result on Windows, turn on `Settings > Privacy & security > Location` before running the app.
5. You can still force a specific location by setting `ACCIDENT_PC_LAT`, `ACCIDENT_PC_LON`, and optionally `ACCIDENT_PC_PLACE`, `ACCIDENT_PC_CITY`, `ACCIDENT_PC_REGION`, `ACCIDENT_PC_COUNTRY`.
