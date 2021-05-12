# cs674_final_project
Apply KPConv (Kernel point convolution) for the task of shape segmentation based on the ShapeNet-Part dataset

## Contribution
The project is forked and developed based on [HuguesTHOMAS/KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

### Gary Wei
TODO

### Genglin Liu
TODO

## Installation
Please refer to [INSTALL.md](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/INSTALL.md) and [setup_env.sh](setup_env.sh) to install all dependencies.
1.  Make sure [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) are installed. One configuration has been tested:
    * PyTorch 1.8.1, CUDA 11.0 and cuDNN 8.0
2. Ensure all python packages are installed
```
sudo apt update
sudo apt install python3-dev python3-pip python3-tk python3-virtualenv
```

3. (Optional) Make a pip, or conda(not recommended), virtual environment
```
python -m venv .venv
. ./.venv/bin/activate
```

4. Follow [PyTorch installation procedure](https://pytorch.org/get-started/locally/)
5. To do the training and testing task, install the following dependencies
   * numpy
   * scikit-learn
   * PyYAML
   * python-dotenv
6. To also do the visualization task, install
   * matplotlib
   * mayavi (not compatible with `conda`, so `pip` virtual environment is recommended)
   * PyQt5
   * open3d
   * jupyter
7. Compile the C++ extension modules for python located in cpp_wrappers. Open a terminal in this folder, and run:
```
cd src/cpp_wrappers
sh compile_wrappers.sh
```

You should now be able to train Kernel-Point Convolution models


### Notes
* Only works on Linux, encounter strange type issue when running on Windows 10.
* Tested on Ubuntu 20.04, RTX 2060(6G), CUDA 11.0 + cuDNN 8.1.
* Recommend at least 32G RAM memory and 16G GPU memory to perform all tasks. Works well with a single Nvidia Tesla T4 GPU (AWS ec2 g4dn.2xlarge instance)


## Prepare the datasets
1. Follow [setup_aws.sh](setup_aws.sh) to download the dataset.
```
wget --no-check-certificate https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
```

2. ***(Important)*** Make a `.env` to declare the environment variables for the path of the dataset. A `.env.template` is provided in the repo.

## Train the model
Please refer to [scene_segmentation_guide.md](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/doc/scene_segmentation_guide.md) and [run.sh](run.sh) for details.
### To repeat the S3DIS Scene Segmentation task
```
python3 -m src.models.train_s3dis
```

### To do the Segmentation task on ShapeNet-Part Dataset
```
python3 -m src.models.train_shapnet_part --ctg <category> --in-radius <in_radius> --feature-dim <in_features_dim>
```

where
* `<category>`: which category of the objects to be trained on, *Airplane*, *Car*, *Chair*, or *Lamp*, default *Airplane*
* `<in_radius>`: Radius of the input sphere, default *0.15*
* `<in_features_dim>`: dimension of the input feature, *1*, *4*, or *5*, default *4*. Refer to the report for more details.

### To repeat the architecture modification experiments
Manually change [`src/models/train_shapnet_part.py`](src/models/train_shapnet_part.py#L34) line 34 to
```
from src.models.architectures_alternative_losses import KPFCNN
```

Then follow the instruction above to train the model.

## Test the model
Similar to the original source code implemented in [HuguesTHOMAS/KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch), we use the same validation set and testing set. So an easy way to see the validation result in to add the path of `previous_training_path` in [`src/models/train_shapnet_part.py`](src/models/train_shapnet_part.py#L220) line 220 and run the training scripts.

The testing scripts provided by [HuguesTHOMAS/KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) is to perform a voting test, which means it takes really a long time before the test finishes if the performance of the model isn't ideal.

### To perform the voting test
1. Change the checkpoint path at [`src/test/test_models.py`](src/test/test_models.py#L98) line 98.
2. In [`src/test/test_models.py`](src/test/test_models.py), you will find detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script:
```
python3 -m src.test.test_models
```


## Plot the results
When you start a new training, it is saved in a `results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model checkpoints, etc.

In [`plot_convergence.py`](src/visualization/plot_convergence.py), you will find detailed comments explaining how to choose which training log you want to plot. Follow them and then run the script :
```
python3 -m src.visualization.plot_convergence
```


## References
* [KPConv: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/pdf/1904.08889.pdf)
* [HuguesTHOMAS/KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch)
* [KPConv：点云核心点卷积 (ICCV 2019)](https://zhuanlan.zhihu.com/p/92244933)


## Structure of the project
Please refer to the document of [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#starting-a-new-project) for the structure of the project.
