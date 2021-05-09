# cs674_final_project
3D Part Segmentation on PartNet

Apply KPConv (Kernel point convolution) for the task of shape segmentation based on the PartNet dataset


## Installation
Only works on Linux, encounter strange type issue when running on Windows 10.

Tested on Ubuntu 20.04, RTX 2060(6G), CUDA 11.0 + cuDNN 8.1


## TODO
- Get PartNet dataset from https://shapenet.org/download/parts
- script to load dataset
- Model architecture (maybe focus on just the rigid one for now)
- train / evaluation code
- experiments to achieve the same performance presented in the paper
- write up report in latex


## References
* [KPConv: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/pdf/1904.08889.pdf)
* [HuguesTHOMAS/KPConv](https://github.com/HuguesTHOMAS/KPConv)
* [KPConv：点云核心点卷积 (ICCV 2019)](https://zhuanlan.zhihu.com/p/92244933)


## Structure of the project
Please refer to the document of [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#starting-a-new-project) for the structure of the project.
