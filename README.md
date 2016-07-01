### Disclaimer

The official Faster R-CNN code (written in MATLAB) is available [here](https://github.com/ShaoqingRen/faster_rcnn).
If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn).

This repository contains a Python *reimplementation* of the MATLAB code.
This Python implementation is built on a fork of [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn).
There are slight differences between the two implementations.
In particular, this Python port
 - is ~10% slower at test-time, because some operations execute on the CPU in Python layers (e.g., 220ms / image vs. 200ms / image for VGG16)
 - gives similar, but not exactly the same, mAP as the MATLAB version
 - is *not compatible* with models trained using the MATLAB code due to the minor implementation differences
 - **includes approximate joint training** that is 1.5x faster than alternating optimization (for VGG16) -- see these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more information

# This repo

This repo is a fork of `rbgirshick/py-faster-rcnn` which allows using Py-Faster-RCNN on Windows. Its branch `windows` was forked from `rbgirshick/py-faster-rcnn`'s master branch on 20 May 2016, and contains minimal modifications that allow running training, evaluation, and deployment on Windows.

This README file was changed accordingly to reflect the changes in the installation and usage under Windows.

_Note about Caffe:_ In this repo I use a different version of Caffe than [`rbgirshick/caffe-fast-rcnn`](https://github.com/rbgirshick/caffe-fast-rcnn). I use [`kukuruza/caffe-fast-rcnn`](https://github.com/kukuruza/win-caffe-fast-rcnn), which is essentially the `faster-rcnn` branch of `rbgirshick/caffe-fast-rcnn` merged into the master branch of [`MSRCCS/caffe`](https://github.com/MSRCCS/caffe) repo. The merge was made in May 20, 2016. Therefore, an advantage of this repo over the original `rbgirshick/py-faster-rcnn` is that a newer version of Caffe is used. See [the installation section](#installation-sufficient-for-the-demo) for details.

The work has been done while I was was a Microsoft Research intern in the [Cloud Computing and Storage](http://research.microsoft.com/en-us/UM/redmond/groups/ccs/) group.

# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)

The original Python implementation contains contributions from Sean Bell (Cornell) written during an MSR internship.

Please see the official [README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more details.

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497) and was subsequently published in NIPS 2015.

### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
2. Python 2.7+
3. Python packages you might not have: `cython`, `python-opencv`, `easydict`
4. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo). 

Type commands into [Powershell](https://msdn.microsoft.com/en-us/powershell/mt173057.aspx).

1. Clone the Faster R-CNN repository, and checkout the required branch
    ```PowerShell
    git clone https://github.com/kukuruza/py-faster-rcnn
    chdir py-faster-rcnn
    git checkout ccs
    ```
    We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`. Note that Caffe is not a submodule any longer.
  
2. Clone `caffe-fast-rcnn` repo inside `FRCN_ROOT` and checkout the required branch.
   (Alternatively you can clone `caffe-fast-rcnn` into a different location and create a symbolic link.)
    ```PowerShell
    chdir $FRCN_ROOT\..
    git clone https://github.com/kukuruza/Caffe-CCS 
    mv Caffe-CCS caffe-fast-rcnn
    chdir caffe-fast-rcnn
    git checkout ccs-faster-rcnn
    ```

3. Build the Cython modules
    ```PowerShell
    chdir $FRCN_ROOT\lib
    python setup.py build_ext --inplace
    cmd /c rmdir build /s /q
    ```

4. Build Caffe and pycaffe
   
  Follow the instructions from https://github.com/MSRCCS/Caffe. 
  Make sure to compile `WITH_PYTHON_LAYER` and build `pycaffe`.
  It is also recommended that you use CUDNN.

5. Download pre-computed Faster R-CNN detectors
    ```PowerShell
    chdir $FRCN_ROOT
    python data\scripts\fetch_faster_rcnn_models.py
    ```

    This will populate the `$FRCN_ROOT\data` folder with `faster_rcnn_models`. See `data\README.md` for details.
    These models were trained on VOC 2007 trainval.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```PowerShell
chdir $FRCN_ROOT
python tools\demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```PowerShell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

3. It should have this basic structure

	```PowerShell
  	$VOCdevkit\                           # development kit
  	$VOCdevkit\VOCcode\                   # VOC utility code
  	$VOCdevkit\VOC2007\                   # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset (need to run PowerShell as administrator)

	```PowerShell
	chdir $FRCN_ROOT\data
	cmd /c mklink -d $VOCdevkit VOCdevkit2007
	```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] If you want to use COCO, please see some notes under `data\README.md`
7. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```PowerShell
chdir $FRCN_ROOT
python data\scripts\fetch_imagenet_models.py
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments\scripts\faster_rcnn_alt_opt.py`.
Output is written underneath `$FRCN_ROOT\output`.

```PowerShell
chdir $FRCN_ROOT
python experiments\scripts\faster_rcnn_alt_opt.py [--GPU gpu_id] [--NET net] [...]
# GPU is the GPU id you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# last optional arguments allow you to specify fast_rcnn.config options, e.g.
#   EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments\scripts\faster_rcnn_end2end.py`.
Output is written underneath `$FRCN_ROOT\output`.

```PowerShell
chdir $FRCN_ROOT
python experiments\scripts\faster_rcnn_end2end.py [--GPU gpu_id] [--NET net] [...]
# GPU is the GPU id you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# last optional arguments allow you to specify fast_rcnn.config options, e.g.
#   EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output\<experiment directory>\<dataset name>\
```

Test outputs are saved under:

```
output\<experiment directory>\<dataset name>\<network snapshot name>\
```
