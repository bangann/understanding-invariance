## Understanding the Generalization Benefit of Model Invariance from a Data Perspective
This is the code for our NeurIPS2021 paper "Understanding the Generalization Benefit of Model Invariance from a Data Perspective". 
There are two major parts in our code: sample covering number estimation and generalization benefit evaluation.

## Requirments
* Python 3.8
* PyTorch
* torchvision
* scikit-learn-extra
* scipy
* robustness package (already included in our code)

Our code is based on [robustness package](https://github.com/MadryLab/robustness).

## Dataset
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) Download and extract the data into /data/cifar10
* [R2N2](http://3d-r2n2.stanford.edu/) Download the ShapeNet rendered images and put the data into /data/r2n2

The randomly sampled R2N2 images used for computing sample covering numbers and indices of examples for different sample sizes could be found [here](https://drive.google.com/file/d/1gT_GNk3QCX56LUHW5HXLF_qs0d8yTs_X/view?usp=sharing).


## Estimation of sample covering  numbers
To estimate the sample covering numbers of different data transformations, run the following script in /scn.
```
CUDA_VISIBLE_DEVICES=0 python run_scn.py  --epsilon 3 --transformation crop --cover_number_method fast --data-path /path/to/dataset 
```
Note that the input is a N x C x H x W tensor where N is sample size.

## Evaluation of generalization benefit
To train the model with data augmentation method, run the following script in /learn_invariance for R2N2 dataset
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset r2n2 \
    --data ../data/2n2/ShapeNetRendering \
    --metainfo-path ../data/r2n2/metainfo_all.json \
    --transforms view  \
    --inv-method aug \
    --out-dir /path/to/out_dir \
    --arch resnet18 --epoch 110 --lr 1e-2 --step-lr 50 \
    --workers 30 --batch-size 128 --exp-name view
```
or the following script for CIFAR-10 dataset
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset cifar \
    --data ../data/cifar10 \
    --n-per-class all \
    --transforms crop  \
    --inv-method aug \
    --out-dir /path/to/out_dir \
    --arch resnet18 --epoch 110 --lr 1e-2 --step-lr 50 \
    --workers 30 --batch-size 128 --exp-name crop 
```
By setting --transforms to be one of {none, flip, crop, rotate, view}, the specific transformation will be considered.


To train the model with regularization method, run the following script. Currently, the code only support 3d-view transformation on R2N2 dataset.
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset r2n2 \
    --data ../data/r2n2/ShapeNetRendering \
    --metainfo-path ../data/r2n2/metainfo_all.json \
    --transforms view  \
    --inv-method reg \
    --inv-method-beta 1 \
    --out-dir /path/to/out_dir \
    --arch resnet18 --epoch 110 --lr 1e-2 --step-lr 50 \
    --workers 30 --batch-size 128 --exp-name reg_view 
```
To evaluate the model with invariance loss and worst-case consistency accuracy, run the following script.
```
CUDA_VISIBLE_DEVICES=0 python main.py  \
    --dataset r2n2 \
    --data ../data/r2n2/ShapeNetRendering \
    --metainfo-path ../data/r2n2/metainfo_all.json \
    --inv-method reg \
    --arch resnet18 \
    --resume /path/to/checkpoint.pt.best \
    --eval-only 1 \
    --transforms view  \
    --adv-eval 0 \
    --batch-size 2  \
    --no-store 
```
Note that to have the worst-case consistency accuracy we need to load 24 view images in R2N2RenderingsTorch class in dataset_3d.py.