# V2V-PoseNet-pytorch
This is a pytorch implementation of V2V-PoseNet([V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map](https://arxiv.org/abs/1711.07399)), which is largely based on the author's [torch7 implementation](https://github.com/mks0601/V2V-PoseNet_RELEASE).

This repository provides
* V2V-PoseNet core modules(model, voxelization, ..)
* An experiment demo on MSRA hand pose dataset, result in ~11mm mean error.
* *Additional [Integral Pose Loss](https://arxiv.org/abs/1711.08229) (or [PoseFix Loss](https://arxiv.org/abs/1812.03595)) implementation*, result in ~10mm mean error on the same demo.

## Requirements
* pytorch 0.4.1 or pytorch 1.0
* python 3.6
* numpy

### **Warning on pytorch0.4.1 cudnn**:
May need to **disable cudnn for batchnorm**, or just only use cuda instead. With cudnn for batchnorm and in float precision, the model cannot train well. My simple experiments show that:

```
cudnn+float: NOT work(e.g. the loss decreases much slower, and result in a higher loss) 
cudnn+float+(disable batchnorm's cudnn): work(e.g. the loss decreases faster, and result in a lower loss)
cudnn+double: work, but the speed is slow
cuda+(float/double): work, but uses much more memroy
```

There is a similar issue pointed out by https://github.com/Microsoft/human-pose-estimation.pytorch. As suggested, disable cudnn for batchnorm:

```
PYTORCH=/path/to/pytorch
for pytorch v0.4.0
sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
for pytorch v0.4.1
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
```

## MSRA hand dataset demo
### Usage
- Clone this repo:
```
git clone https://github.com/dragonbook/V2V-PoseNet-pytorch.git
cd V2V-PoseNet-pytorch
```

- Download [MSRA hand dataset](https://jimmysuen.github.io/) and extract to directory path/to/msra-hand.

- Download [estimated centers](https://cv.snu.ac.kr/research/V2V-PoseNet/MSRA/center/center.tar.gz) of MSRA hand dataset which required by V2V-PoseNet and provided by the [author's implementation](https://github.com/mks0601/V2V-PoseNet_RELEASE). Extract them to the directory path/to/msra-hand-center. 
```
Note, this repository contains a copy of the msra hand centers under ./datasets/msra_center.
```

- Configure data_dir=path/to/msra-hand and center_dir=path/to/msra-hand-center in ./experiments/msra-subject3/main.py. And Run following command to perform training and testing. It will train the dataset for few epochs and evaluate on the test dataset. The test result will be saved as test_res.txt and the fit result on training data will be saved as fit_res.txt
```
PYTHONPATH=./ python ./experiments/msra-subject3/main.py
```

- Configure data_dir=path/to/msra-hand and center_dir=path/to/msra-hand-center in ./experiments/msra-subject3/gen_gt.py. Run it to generate ground truth labels as train_s3_gt.txt and test_s3_gt.txt

- Configure pred_file=path/to/test_s3_gt.txt and gt_file=path/to/test_res.txt in ./experiments/msra-subject3/show_acc.py. Run it to plot accuracy and error.

- The following figures show that the simple experiment can result in about 11mm mean error.

![msra_s3_acc](/figs/msra_s3_joint_acc.png)

![msra_s3_mean_error](/figs/msra_s3_joint_mean_error.png)


## Additional [IntegralPose](https://arxiv.org/abs/1711.08229)/[PoseFix](https://arxiv.org/abs/1812.03595) style loss implementation
Replaced V2V-PoseNet's loss with PoseFix's loss(one-hot heatmap loss + L1 coord loss), and it's independently implemented under ./integral-pose directory. Also, configure data_dir and center_dir in ./integral-pose/main.py, and start training. The result shows about 10mm mean error.

![integral_loss_s3_acc](/figs/integral_pose_msra_s3_joint_acc.png)

![integral_loss_mean_error](/figs/integral_pose_msra_s3_joint_mean_error.png)

![compare_mean_error](/figs/mean_error_compare.png)

# Below is from author's README for reference
# V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map

# Introduction

This is our project repository for the paper, **V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map ([CVPR 2018](http://cvpr2018.thecvf.com))**.

We, **Team SNU CVLAB**, (<i>Gyeongsik Moon, Juyong Chang</i>, and <i>Kyoung Mu Lee</i> of [**Computer Vision Lab, Seoul National University**](https://cv.snu.ac.kr/)) are **winners** of [**HANDS2017 Challenge**](http://icvl.ee.ic.ac.uk/hands17/challenge/) on frame-based 3D hand pose estimation.



Please refer to our paper for details.

If you find our work useful in your research or publication, please cite our work:

[1] Moon, Gyeongsik, Ju Yong Chang, and Kyoung Mu Lee. **"V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map."** <i>CVPR 2018. </i> [[arXiv](https://arxiv.org/abs/1711.07399)]
  
  ```
@InProceedings{Moon_2018_CVPR_V2V-PoseNet,
  author = {Moon, Gyeongsik and Chang, Juyong and Lee, Kyoung Mu},
  title = {V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```

In this repository, we provide
* Our model architecture description (V2V-PoseNet)
* HANDS2017 frame-based 3D hand pose estimation Challenge Results
* Comparison with the previous state-of-the-art methods
* Training code
* Datasets we used (ICVL, NYU, MSRA, ITOP)
* Trained models and estimated results
* 3D hand and human pose estimation examples


## Model Architecture

![V2V-PoseNet](/figs/V2V-PoseNet.png)

## HANDS2017 frame-based 3D hand pose estimation Challenge Results

![Challenge_result](/figs/Challenge_result.png)


## Comparison with the previous state-of-the-art methods

![Paper_result_hand_graph](/figs/Paper_result_hand_graph.png)

![Paper_result_hand_table](/figs/Paper_result_hand_table.png)

![Paper_result_human_table](/figs/Paper_result_human_table.png)

# About our code
## Dependencies
* [Torch7](http://torch.ch)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)

Our code is tested under Ubuntu 14.04 and 16.04 environment with Titan X GPUs (12GB VRAM).

## Code
Clone this repository into any place you want. You may follow the example below.
```bash
makeReposit = [/the/directory/as/you/wish]
mkdir -p $makeReposit/; cd $makeReposit/
git clone https://github.com/mks0601/V2V-PoseNet_RELEASE.git
```
* `src` folder contains lua script files for data loader, trainer, tester and other utilities.
* `data` folder contains data converter which converts image files to the binary files.

To train our model, please run the following command in the `src` directory:

```bash
th rum_me.lua
```

* There are some optional configurations you can adjust in the config.lua. 
* You have to convert the `.png` images of the ICVL and NYU dataset to the `.bin` files by running the code from `data` folder.
* The directory where you have to put the dataset files and computed centers of each frame is defined in `src/data/dataset_name/data.lua`
* Visualization code is finally uploaded! You have to prepare 'result_pixel.txt' for each dataset. Each row of the result file has to contain the pixel coordinates of x, y and depth of all joints (i.e, x1 y1 z1 x2 y2 z2 ...). Then run pixel2world script and run draw_DB.m

# Dataset
We trained and tested our model on the four 3D hand pose estimation and one 3D human pose estimation datasets.

* ICVL Hand Poseture Dataset [[link](https://labicvl.github.io/hand.html)] [[paper](http://www.iis.ee.ic.ac.uk/dtang/cvpr_14.pdf)]
* NYU Hand Pose Dataset [[link](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm)] [[paper](https://cims.nyu.edu/~tompson/others/TOG_2014_paper_PREPRINT.pdf)]
* MSRA Hand Pose Dataset [[link](https://jimmysuen.github.io/)] [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf)]
* HANDS2017 Challenge Dataset [[link](http://icvl.ee.ic.ac.uk/hands17/challenge/)] [[paper](https://arxiv.org/abs/1712.03917)] [[challenge benchmark paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yuan_Depth-Based_3D_Hand_CVPR_2018_paper.pdf)]
* ITOP Human Pose Dataset [[link](https://www.albert.cm/projects/viewpoint_3d_pose/)] [[paper](https://arxiv.org/abs/1603.07076)]


# Results
Here we provide the precomputed centers, estimated 3D coordinates and pre-trained models.

The precomputed centers are obtained by training the hand center estimation network from [DeepPrior++ ](https://arxiv.org/pdf/1708.08325.pdf). Each line represents 3D world coordinate of each frame.
In case of ICVL, NYU, MSRA dataset, if depth map is not exist or not contain hand, that frame is considered as invalid.
In case of ITOP dataset, if 'valid' variable of a certain frame is false, that frame is considered as invalid.
All test images are considered as valid.

The 3D coordinates estimated on the ICVL, NYU and MSRA datasets are pixel coordinates and the 3D coordinates estimated on the ITOP datasets are world coordinates. The estimated results are from ensembled model. You can make the results from a single model by downloading the pre-trained model and testing it.

* ICVL Hand Poseture Dataset [[center_trainset](https://cv.snu.ac.kr/research/V2V-PoseNet/ICVL/center/center_train_refined.txt)] [[center_testset](https://cv.snu.ac.kr/research/V2V-PoseNet/ICVL/center/center_test_refined.txt)] [[estimation](https://cv.snu.ac.kr/research/V2V-PoseNet/ICVL/coordinate/result.txt)] [[models](https://cv.snu.ac.kr/research/V2V-PoseNet/ICVL/model/model.tar.gz)]
* NYU Hand Pose Dataset [[center_trainset](https://cv.snu.ac.kr/research/V2V-PoseNet/NYU/center/center_train_refined.txt)] [[center_testset](https://cv.snu.ac.kr/research/V2V-PoseNet/NYU/center/center_test_refined.txt)] [[estimation](https://cv.snu.ac.kr/research/V2V-PoseNet/NYU/coordinate/result.txt)] [[models](https://cv.snu.ac.kr/research/V2V-PoseNet/NYU/model/model.tar.gz)]
* MSRA Hand Pose Dataset [[center](https://cv.snu.ac.kr/research/V2V-PoseNet/MSRA/center/center.tar.gz)] [[estimation](https://cv.snu.ac.kr/research/V2V-PoseNet/MSRA/coordinate/result.txt)] [[models](https://cv.snu.ac.kr/research/V2V-PoseNet/MSRA/model/model.tar.gz)]
* ITOP Human Pose Dataset (front-view) [[center_trainset](https://cv.snu.ac.kr/research/V2V-PoseNet/ITOP_front/center/center_train.txt)] [[center_testset](https://cv.snu.ac.kr/research/V2V-PoseNet/ITOP_front/center/center_test.txt)] [[estimation](https://cv.snu.ac.kr/research/V2V-PoseNet/ITOP_front/coordinate/result.txt)] [[models](https://cv.snu.ac.kr/research/V2V-PoseNet/ITOP_front/model/model.tar.gz)]
* ITOP Human Pose Dataset (top-view) [[center_trainset](https://cv.snu.ac.kr/research/V2V-PoseNet/ITOP_top/center/center_train.txt)] [[center_testset](https://cv.snu.ac.kr/research/V2V-PoseNet/ITOP_top/center/center_test.txt)] [[estimation](https://cv.snu.ac.kr/research/V2V-PoseNet/ITOP_top/coordinate/result.txt)] [[models](https://cv.snu.ac.kr/research/V2V-PoseNet/ITOP_top/model/model.tar.gz)]

We used [awesome-hand-pose-estimation ](https://github.com/xinghaochen/awesome-hand-pose-estimation) to evaluate the accuracy of the V2V-PoseNet on the ICVL, NYU and MSRA dataset.

Belows are qualitative results.
![result_1](/figs/result/Paper_result_ICVL.png)
![result_2](/figs/result/Paper_result_NYU.png)
![result_3](/figs/result/Paper_result_MSRA.png)
![result_4](/figs/result/Paper_result_HANDS2017.png)
![result_5](/figs/result/Paper_result_ITOP_front.png)
![result_6](/figs/result/Paper_result_ITOP_top.png)
