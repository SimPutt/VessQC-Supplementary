# VessQC-Supplementary: Uncertainty Map Generation for 3D Segmentation

This repository provides the necessary code and workflows to generate **pixel-wise** and **topology-aware** uncertainty maps for 3D volumetric datasets, particularly focused on vessel segmentation.

It supports the research presented in the paper that introduced **VessQC**, an open-source tool for uncertainty-guided curation of large 3D microscopy segmentations. By using the models and scripts within this repository, researchers can reproduce the uncertainty maps required to utilize the full capabilities of the VessQC curation framework.

---

## Overview and Navigation

This supplementary repository is structured to separate the two types of uncertainty estimation. To begin, navigate to the relevant subfolder and follow its specific setup and execution instructions:

| Focus | Description | Navigation |
| :--- | :--- | :--- |
| **Topology-Aware Uncertainty** | Inference pipeline for computing **structure-wise (topology-aware) uncertainty**. | **[`Topology-Aware-Uncertainty`](Topology-Aware-Uncertainty/)** |
| **Pixel-wise Uncertainty** | Scripts and models for generating standard **pixel-wise uncertainty** maps. | **[`Pixel-Wise-Uncertainty`](Pixel-Wise-Uncertainty/)** |
---

## Key Links and Resources

| Resource | Description | URL |
| :--- | :--- | :--- |
| **VessQC Curation Tool** | The main open-source tool that utilizes these uncertainty maps for 3D segmentation curation. | [`https://github.com/MMV-Lab/VessQC`](https://github.com/MMV-Lab/VessQC) |
| **VessQC Publication** | Paper introducing the VessQC tool and its evaluation. | [`Preprint`](https://arxiv.org/pdf/2511.22236)  |
| **Uncertainty Publication** | Paper detailing the methodology for the topology-aware uncertainty component. | [`Paper`](https://proceedings.neurips.cc/paper_files/paper/2023/file/19ded4cfc36a7feb7fce975393d378fd-Paper-Conference.pdf)|

## Citation

If you use the VessQC tool or the methodologies supported by this supplementary repository, please cite the corresponding papers:

### VessQC
```
@article{puttmann2025bridging,
  title={Bridging 3D Deep Learning and Curation for Analysis and High-Quality Segmentation in Practice},
  author={P{\"u}ttmann, Simon and Contreras, Jonathan Jair S{\`a}nchez and Kowitz, Lennart and Lampen, Peter and Gupta, Saumya and Panzeri, Davide and Hagemann, Nina and Xiong, Qiaojie and Hermann, Dirk M and Chen, Cao and others},
  journal={arXiv preprint arXiv:2511.22236},
  year={2025}
}
```


### Topology-Aware Uncertainty
```
@article{gupta2023topology,
  title={Topology-aware uncertainty for image segmentation},
  author={Gupta, Saumya and Zhang, Yikai and Hu, Xiaoling and Prasanna, Prateek and Chen, Chao},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={8186--8207},
  year={2023}
}
```

### Pixel-Wise Uncertainty
TODO

## Contact

For any issues regarding **VessQC** or general inquiries related to this supplementary repository, please email **simon.puettmann@isas.de**.

For any issues regarding **topology-aware uncertainty** calculation or the **DIPHA** dependency, please email **saumgupta@cs.stonybrook.edu**.

For any issues regarding **pixel-wise uncertainty** calculation, please email **jair.sanchez@isas.de**.
