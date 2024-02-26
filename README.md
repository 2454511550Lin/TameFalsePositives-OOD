# Taming False Positives in Out-of-Distribution Detection with Human Feedback
### [ [Paper]](https://harit7.github.io/assets/pdf/ood-fpr-draft.pdf)
<br/>

This is the official repository for Taming False Positives in Out-of-Distribution Detection with Human Feedback. 

## Quick Start

Install the environment by `conda env create -f environment.yml`.

You can run any experiments by setting the correct configuration script. configuration scripts are located at `configs`. For example, to run the cifar10 experiment with change detection:

`bash run.sh cifar10_change.yaml`

Make sure that the `configs/cifar10_change.yaml` exists in the directory (which we have already provided). The log and results are stored in the `output` folder, and the plots are stored in the `plot` folder.

## Folders 

`configs`: this folder contains the parameters configuration of the experiments, including the mode, the importance sampling rate, the window size, etc.

`score`: we have included all the OOD scores we used in the paper, including CIFAR-10, CIFAR-100, MNIST, SVHN, Texture, TinyImageNet, and Places365 datasets

`output`: by default, the result would be stored as `.pkl` files here.

`plot`: by default, the plots of the result would be stored here.
