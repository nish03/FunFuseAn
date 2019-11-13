![GitHub Logo](/docs/logo.png)
This is the open source toolkit for the MBIA-MICCAI 2019 paper titled [Structural Similarity based Anatomical and Functional Brain Imaging Fusion](http://arxiv.org/abs/1908.03958) by **N.Kumar et al**. 

![GitHub Logo](/docs/architecture.png)

The paper performs the following tasks:
* Develops a novel end-to-end unsupervised fusion algorithm for anatomical and functional imaging modalities using Convolutional Neural Networks.
* Develops a mathematical framework to visualise the fused image using the partial derivatives with respect to the input imaging modalities.
* Compares the proposed fusion + visualisation method with other state-of-the-art fusion methods.

**Note**: Please cite the paper if you are using this code in your research.


## Prerequisites
* Python 2.7
* Tensorflow 1.0.x
* numpy
* matplotlib
* skimage
* MATLAB

## Results
![Logo](https://github.com/nish03/FunFuseAn/blob/master/docs/Visual%20results.png)

![Logo1](https://github.com/nish03/FunFuseAn/blob/master/docs/Loss%20curves.png)

## Usage
Coming Soon

## How to Cite
@InProceedings{10.1007/978-3-030-33226-6_14,
author="Kumar, Nishant
and Hoffmann, Nico
and Oelschl{\"a}gel, Martin
and Koch, Edmund
and Kirsch, Matthias
and Gumhold, Stefan",
editor="Zhu, Dajiang
and Yan, Jingwen
and Huang, Heng
and Shen, Li
and Thompson, Paul M.
and Westin, Carl-Fredrik
and Pennec, Xavier
and Joshi, Sarang
and Nielsen, Mads
and Fletcher, Tom
and Durrleman, Stanley
and Sommer, Stefan",
title="Structural Similarity Based Anatomical and Functional Brain Imaging Fusion",
booktitle="Multimodal Brain Image Analysis and Mathematical Foundations of Computational Anatomy",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="121--129",
abstract="Multimodal medical image fusion helps in combining contrasting features from two or more input imaging modalities to represent fused information in a single image. One of the pivotal clinical applications of medical image fusion is the merging of anatomical and functional modalities for fast diagnosis of malign tissues. In this paper, we present a novel end-to-end unsupervised learning based Convolutional neural network (CNN) for fusing the high and low frequency components of MRI-PET grayscale image pairs publicly available at ADNI by exploiting Structural Similarity Index (SSIM) as the loss function during training. We then apply color coding for the visualization of the fused image by quantifying the contribution of each input image in terms of the partial derivatives of the fused image. We find that our fusion and visualization approach results in better visual perception of the fused image, while also comparing favorably to previous methods when applying various quantitative assessment metrics.",
isbn="978-3-030-33226-6"
}
