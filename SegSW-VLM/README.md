# SegSW-VLM

Official research code for **SegSW-VLM: a vision-language foundation model for solid waste segmentation across 264 global cities**.


## Abstract

SegSW-VLM is a multimodal vision-language framework designed for precise solid waste segmentation from very-high-resolution remote sensing imagery. It jointly leverages satellite images and GeoText descriptions to improve the semantic understanding and spatial localization of solid waste dumps in complex urban and natural environments.
This repository provides the implementation of SegSW-VLM together with training, inference, and evaluation scripts for solid waste segmentation. The model is built upon a multimodal large language model branch and a segmentation foundation model branch. To better adapt vision-language segmentation to remote sensing solid waste scenes, SegSW-VLM introduces two key components: Attribute-Prior Semantic Alignment (APSA), which incorporates structured solid waste attributes into semantic query learning, and Evidence-Prior Alignment Adapter (EPAA), which converts waste-aware semantic queries into spatial prompts for accurate mask decoding.
The model is developed based on GSWSeg4MM, a global multimodal solid waste segmentation dataset containing about 103k image-text samples collected from 264 cities worldwide. Each sample includes a VHR satellite image, a pixel-level solid waste mask, and GeoText descriptions covering waste type, scene context, color, shape, boundary, density, and surrounding environment.


## Method Overview

SegSW-VLM follows a LLaVA + SAM style vision-language segmentation architecture, but introduces two task-specific mechanisms for remote-sensing solid-waste segmentation:

1. **APSA: Attribute-Prior Semantic Alignment**
   - `SAP`: parses structured GeoText attributes such as waste type, scene type, shape form, color, boundary, and density.
   - `SWQ-Former`: learns solid-waste semantic queries from multimodal hidden states and attribute priors.

2. **EPAA: Evidence-Prior Alignment Adapter**
   - `HECG`: constructs a hierarchical evidence field from multi-scale SAM image embeddings.
   - `CAPG`: generates contrast-aware prompts, including target, boundary, and distractor prompts, to guide SAM mask decoding.


## Installation

```
conda create -n segswvlm python=3.10 -y
conda activate segswvlm
pip install -r requirements.txt
```

The code was developed with PyTorch, DeepSpeed, LLaVA, and SAM. Please prepare the following checkpoints before training or inference:

- LLaVA-v1.5-7B or a compatible LLaVA checkpoint.
- SAM ViT-H checkpoint, e.g. `sam_vit_h_4b8939.pth`.


## Dataset Format

The expected dataset layout is:
```text
GSWSeg4MM/
├── train/
│   ├── img/                 
│   ├── lbl/                 
│   └── json/                
├── val/
│   ├── img/                 
│   ├── lbl/
│   └── json/
├── val/
    ├── img/                 
    ├── lbl/
    └── json/
```


## Training

Set paths :

```
export LLAVA_PATH=/path/to/llava-v1.5-7b
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
export DATA_ROOT=/path/to/GSWSeg4MM
```

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{segswvlm2026,
  title   = {SegSW-VLM: A Vision-Language Foundation Model for Solid Waste Segmentation across 264 Global Cities},
  author  = {Chengxuan Zhao, Quanlong Feng, Shiying Yuan, Bowen Niu, Yinghui Sun, Yelu Zeng, Ying Li, Shoujing Yin, Jiaqi Yu,Yimin Wang, Jianhua Gong, Gaofei Yin, Jianxi Huang, Jiantao Liu},
  journal = {Under review},
  year    = {2026}
}
```

## Acknowledgements

This project builds upon LLaVA, SAM, and LISA. We thank the open-source community for making these foundations available.


