# Make Interval Bound Propagation great again

## Abstract
In various scenarios motivated by real life, such as medical data analysis, autonomous driving, and adversarial training, we are interested in robust deep networks. A network is robust when a relatively small perturbation of the input cannot lead to drastic changes in output (like change of class, etc.). This falls under the broader scope field of Neural Network Certification (NNC).
Two crucial problems in NNC are of profound interest to the scientific community: how to calculate the robustness of a given pre-trained network and how to construct robust networks. The common approach to constructing robust networks is Interval Bound Propagation (IBP). 
This paper demonstrates that IBP is sub-optimal in the first case due to its susceptibility to the wrapping effect. Even for linear activation, IBP gives strongly sub-optimal bounds. Consequently, one should use strategies immune to the wrapping effect to obtain bounds close to optimal ones. We adapt two classical approaches dedicated to strict computations -- Dubleton Arithmetic and Affine Arithmetic -- to mitigate the wrapping effect in neural networks. These techniques yield precise results for networks with linear activation functions, thus resisting the wrapping effect. As a result, we achieve bounds significantly closer to the optimal level than IBPs.

## Teaser
The Affine Arithmetic (AA) method is able to reduce the wrapping effect compared to the Interval Bound Propagation (IBP) method.

![Working scheme of the AA and IBP methods](./teaser.png)

## Datasets
For the experiments and ablation study, we use 4 publicly available datasets:
* [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
* [CIFAR-10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) 
* [SVHN](https://pytorch.org/vision/main/generated/torchvision.datasets.SVHN.html)
* [Digits](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html)


The datasets may be downloaded when the algorithm runs.

## Commands
**Setup**
```
conda create -n "IBP_great_again" python=3.10
pip install -r requirements.txt
cp example.env .env
edit .env
```

**Launching Experiments**
```
conda activate IBP_great_again
WANDB_MODE={offline/online} HYDRA_FULL_ERROR={0/1} python src/main.py --config-name config 
```