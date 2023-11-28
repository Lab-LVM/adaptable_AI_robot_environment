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