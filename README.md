# CATS
## Introduction
- Federated test-time adaptation (FTTA) allows models to adapt to unlabeled target data during inference while preserving privacy. However, it faces challenges from the dynamic availability of source clients and the uncertainty of test samples under distribution shifts.
- We propose a novel framework that synergistically combines adaptation rate optimization with a robust sample filtering mechanism. Specifically, it integrates adaptation rate optimization during training with a fine-grained personalized thresholding strategy during testing, which leverages CAIN values and a weighted pseudo-labeling mechanism to achieve highly effective sample selection. Furthermore, to prevent model overconfidence, we introduce a min-max entropy regularization mechanism.
 ##  Core Algorithm: Federated learning, test-time adaptation
- This repository contains the official implementation of the CATS framework. The core innovation lies in two collaboratively working modules:
> **Synergistic Adaptation and Filtering:** Integrates adaptation rate optimization during training with a fine-grained personalized thresholding strategy during testing. By leveraging CAIN values and a weighted pseudo-labeling mechanism, it achieves highly effective sample selection.
> >
> **Min-Max Entropy Regularization:**  Dynamically identifies data uncertainty at test-time by perturbing samples with patch shuffling. It applies entropy maximization specifically to uncertain samples, effectively preventing model overconfidence and enabling more reliable test-time adaptation.
> ><img width="590" height="495" alt="6898ecb9-e797-4d8b-aad0-ecce5f893297" src="https://github.com/user-attachments/assets/351ebdac-d0ed-461d-8c5e-eedb82ff3812" />
## Requirements
- python 3.8.5
- cudatoolkit 10.2.89
- cudnn 7.6.5
- pytorch 1.11.0
- torchvision 0.12.0
- numpy 1.18.5
- tqdm 4.65.0
- matplotlib 3.7.1
If you prefer generating the CIFAR-10C, CIFAR-100C and Tiny-ImageNetC by yourself, these packages may also be required:
- wandb 0.16.0
- scikit-image 0.17.2
- opencv-python 4.8.0.74
## Install Datasets
We need users to declare a `data` to store the dataset as well as the log of training procedure. The directory structure should be :

Download the datasets used in our paper from the following links:

- [CIFAR-100](https://flow/file_open?url=https%3A%2F%2Fwww.cs.toronto.edu%2F~kriz%2Fcifar.html&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
- [Tiny-ImageNet](https://flow/file_open?url=http%3A%2F%2Fcs231n.stanford.edu%2Ftiny-imagenet-200.zip&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
- [Food-101](https://flow/file_open?url=https%3A%2F%2Fdata.vision.ee.ethz.ch%2Fcvl%2Fdatasets_extra%2Ffood-101%2F&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
- [PACS](https://flow/file_open?url=https%3A%2F%2Fdomaingeneralization.github.io%2F%23data&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
```
data
│       
└───dataset
│   │   CIFAR100
│       │  test
│       │  train
|       |  meta
│       │  file.txt
│   │   Tiny-ImageNet
│       │  test
│       │  train
│       │  val
│   │   PACS
│       │  art_painting
│       │  cartoon
│       │  photo
│       │  sketch

```
## Run
## CIFAR-100C Experiments
We consider feature shifts in our CIFAR-100C experiments.
## Generate Dataset
```
./data_prepare.sh
```
This shell script will partition the CIFAR-100 dataset to 300 clients (240 source clients and 60 clients), and save the partition indices to `~/data/feddc/partition/cifar100/`.Before running this script, you need to manually set the different corruptions for training and testing.
## Train Global Model
Before running CATS, we need to train a global model with source clients' training sets.
```
./pretrain_fedawi_${model}.sh
```
Here `${model}` specifies the model architecture we use. We used resnet18 (ResNet-18) and resnet50 in our paper.
Learn Adaptation Rates
```
./fed_train_${model}.sh
```
Learn CAIN Thresholds
```
python c_cifar100.py
```
## Fedderated Test-Time Adaptation with CATS-batch and CATS-online
```
./CATS_test_${model}.sh
```
You can run most of the experiments in our paper by  
shell: python main.py

Moreover, we also prepare code for various datasets and model architectures. Please check the arguments function in the `option.py` file for more details.
## Acknowledgements
This implementation is based on [ATP](https://github.com/baowenxuan/ATP) and [[DEYO](https://whitesnowdrop.github.io/DeYO/).
