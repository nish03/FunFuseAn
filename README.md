![GitHub Logo](/docs/logo.png)
This is the open source toolkit for the MBIA-MICCAI 2019 paper titled [Structural Similarity based Anatomical and Functional Brain Imaging Fusion](https://link.springer.com/chapter/10.1007/978-3-030-33226-6_14) by **N.Kumar et al**. 

![GitHub Logo](/docs/architecture.png)

The paper performs the following tasks:
* Develops a novel end-to-end unsupervised fusion algorithm for anatomical and functional imaging modalities using Convolutional Neural Networks.
* Develops a mathematical framework to visualise the fused image using the partial derivatives with respect to the input imaging modalities.
* Compares the proposed fusion + visualisation method with other state-of-the-art fusion methods.

**Note**: Please cite the paper if you are using this code in your research.


## Prerequisites
* Python 2.7
* Tensorflow 1.0.x or Pytorch 1.0 and above
* numpy
* matplotlib
* skimage
* MATLAB

## Results
![Logo](https://github.com/nish03/FunFuseAn/blob/master/docs/Visual%20results.png)

![Logo1](https://github.com/nish03/FunFuseAn/blob/master/docs/Loss%20curves.png)

  ## Data
The training of the FunFuseAn network was done with 500 MRI-PET image pairs available at Alzheimer’s Disease Neuroimaging Initiative (ADNI) as mentioned in the paper. Although, the data is available for public use, it is still required to apply for getting the access to the repository by filling out a questionaire. In case you are interested to obtain the data, please apply at [this link](http://adni.loni.usc.edu/data-samples/access-data/). For conducting the inference of our network, in addition to the ADNI test data, we also used 10 image pairs from [Harvard Whole Brain Atlas](http://www.med.harvard.edu/AANLIB/) which is also publicly available and donot required any written permission. We therefore uploaded this data for your usage.

## How to Cite
@InProceedings{10.1007/978-3-030-33226-6_14,
author="Kumar, Nishant
and Hoffmann, Nico
and Oelschl{\"a}gel, Martin
and Koch, Edmund
and Kirsch, Matthias
and Gumhold, Stefan",
title="Structural Similarity Based Anatomical and Functional Brain Imaging Fusion",
booktitle="Multimodal Brain Image Analysis and Mathematical Foundations of Computational Anatomy",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="121--129",
isbn="978-3-030-33226-6"
}
