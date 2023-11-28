# Open World Object Detection

This project contains Pytorch-based object detection for open world. <br/>
We purpose to identify and characterize unknown instances to assist relieveing confusion within an incremental object detection framework.

## Tutorial

We provide GUI on the web page for convenience of using our methods. <br/>
run `app.py` and access the guided web page. <br/>
If you want to match your settings with our development environment, refer the `requirements.txt`. <br/>

### 1. Real-time Open world object detection

- If you want to use real-time Open world object detection by webcam, press `Start Webcam`.

### 2. Make YAML with new label

- If you want to create new label to the unknown label, upload unkown label.
- and fill in the new label name, press `Download YAML`.
- The label of new name must be **ENGLISH** only.

### 3. Train new model

- If you want to train new model, press `TRAIN 시작`.
- Press `Unknown Label Upload` and `Download YAML` first if you train new model with unknown image.
- Once train is terminated, required to restart app.

- Depending on the user's environment some settings should be modifed appropriately:
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

# Face Domain Adaption

This project contains Pytorch-based domain adaptation for face recognition. We aim to solve two typical real-world face recognition problems: masked face recognition, and corrupted face recognition. To solve this challenging task, we use various domain adaptation methods (DANN, CDAN, BSP, etc).



## Tutorial

We provide a basic command to train/test our method on the masked face recognition dataset (MFR1).

- To see each method's detailed configuration, please see `configs/`.
- To see each dataset's specification, please see `docs/Evaluation.txt`.




1. train

```bash
# checkpoint is saved into pretrained/DABase_IResNet/Unmasked+Synthetic_Masked_backbone.ckpt
python3 main.py fit --config=configs/corrupt/source_only.yaml -g '0,' --seed_everything 2021
```

2. test

```bash
# modify model-init_args-backbone_init-pretrained_path to the trained checkpoint (configs/source_only.yamsl)
python3 main.py test --config=configs/corrupt/source_only.yaml -g '0,' --seed_everything 2021
```



## Experiment setup

| category      | value                             |
| ------------- | --------------------------------- |
| model         | iResNet50                         |
| epochs        | 25                                |
| dropout       | 0.1                               |
| image size    | 112, 112                          |
| learning rate | 0.001                             |
| batch size    | 32                                |
| optimizer     | SGD (momentum=0.9, nesterov=True) |



## Project structure

```bash
# project structure under `/source_free_domdain_adaptation`
/source_free_domain_adaptation
├── LICENSE
├── README.md
├── config.yaml
├── configs
├── data
├── docs
├── main.py
├── save_backbone.py
├── src
└── tmuxp
```



## Dataset structure

```bash
# dataset structure under `/data` folder.
├── jpg
│   ├── MFR1
|   |   ├── Masked
|   |   ├── Synthetic
|   |   ├── Unmasked
|   |   ├── Unmasked+Synthetic
|   |   ├── dataset_info.txt
|   |   ├── pairs
|   |   ├── split.py
|   |   ├── split_class_dict.json
|   |   └── split_class_dict_info.txt
├── test
│   ├── MFR1
│   │   ├── Masked
│   │   ├── Unmasked
│   │   ├── m_m.bin
│   │   ├── m_m_split.bin
│   │   ├── u_m.bin
│   │   ├── u_m_split.bin
│   │   ├── u_u.bin
│   │   └── u_u_split.bin
```



## Requirements

we test this project on the following environments:

- ubuntu linux == 18.04
- python >= 3.7.5
- cuda >= 11.3
- pytorch >= 1.10.0
