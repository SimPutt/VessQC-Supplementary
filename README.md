# VessQC-Supplementary: Uncertainty Map Generation for 3D Segmentation

This repository provides the necessary code and workflows to generate **pixel-wise** and **topology-aware** uncertainty maps for 3D volumetric datasets, particularly focused on vessel segmentation.

It supports the research presented in the paper that introduced **VessQC**, an open-source tool for uncertainty-guided curation of large 3D microscopy segmentations. By using the models and scripts within this repository, researchers can reproduce the uncertainty maps required to utilize the full capabilities of the VessQC curation framework.

---

## Repository Structure

This repository is organized to separate the methods used for generating different types of uncertainty maps:

* **[`topology-uncertainty/`](topology-uncertainty/)**: Contains the inference pipeline for computing **structure-wise (topology-aware) uncertainty** for 3D data. This is based on the work from "Topology-Aware Uncertainty for Image Segmentation".
* **[`pixelwise-uncertainty/`](pixelwise-uncertainty/)**: Contains the scripts and models for generating **pixel-wise uncertainty** maps.

---

## Get Started

To begin generating uncertainty maps, please navigate to the corresponding subfolder and follow its specific setup and execution instructions:

* For **Topology-Aware Uncertainty**, see the instructions in **[`topology-uncertainty/`](topology-uncertainty/)**.
* For **Pixel-wise Uncertainty**, see the instructions in **[`pixelwise-uncertainty/`](pixelwise-uncertainty/)**.

---

## Citation

If you use the VessQC tool or the methodologies supported by this supplementary repository, please cite the corresponding papers:

### VessQC
TODO
<!---
@article{puettmann2025vessqc,
  title={Bridging 3D Deep Learning and Uncertainty-Guided Curation for Analysis and High-Quality Segmentation Ground Truth},
  author={Püttmann, Simon and Sánchez Contreras, Jonathan Jair and Kowitz, Lennart and Lampen, Peter and Gupta, Saumya and Panzeri, Davide and Hagemann, Nina and Xiong, Qiaojie and Hermann, Dirk and Chen, Chao and Chen, Jianxu},
  journal={Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2025}}
```
-->

### Topology-Aware Uncertainty
```
@article{gupta2024topology, 
  title={Topology-aware uncertainty for image segmentation}, 
  author={Gupta, Saumya and Zhang, Yikai and Hu, Xiaoling and Prasanna, Prateek and Chen, Chao}, 
  journal={Advances in Neural Information Processing Systems}, 
  volume={36}, 
  year={2024} }
```

### Pixel-Wise Uncertainty
TODO

## Contact

For any issues regarding **VessQC** or general inquiries related to this supplementary repository, please email **simon.puettmann@isas.de**.

For any issues regarding **topology-aware uncertainty** calculation or the **DIPHA** dependency, please email **saumgupta@cs.stonybrook.edu**.

For any issues regarding **pixel-wise uncertainty** calculation, please email **jair.sanchez@isas.de**.
