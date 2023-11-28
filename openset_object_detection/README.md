# Open World Object Detection

This project contains Pytorch-based object detection for open world. We purpose to identify and characterize unknown instances to assist relieveing confusion within an incremental object detection framework.

## Tutorial

We provide GUI on the web page for convenience of using our methods.
run `app.py` and access the guided web page.
If you want to match your settings with our development environment, refer the `requirements.txt`.

1. Real-time object tracking

- If you want to use real-time object tracking by webcam, press `Start Webcam`.

2. Make YAML with new label

- If you want to create new label to the unknown label, upload unkown label.
- and fill in the new label name, press `Download YAML`.

* The label of new name must be ENGLISH only.

3. Train new model

- If you want to train new model, press `TRAIN 시작`.

* Press `Unknown Label Upload` and `Download YAML` first if you train new model with unknown image.
* Once train is terminated, required to restart app.

* Depending on the user's environment some settings should be modifed appropriately:

- The unknown config and config in flask_detect.py
- The similarconf in app.py

## Project structure

```bash
/openset_object_detection
├── data
├── templates
├── utils
├── app.py
├── benchmarks.py
├── detect.py
├── export.py
├── hubconf.py
├── train.py
├── val.py
├── requirements.txt
├── setup.cfg
└── README.md
```
