## Introduction

A Pytorch Implementation of ResNet(56-layers) trained on Cifar-10 with Accuracy of 92.07%.

- [X] train/evaluate model
- [X] Tensorboard
- [X] CPU/GPU version

## Train

+ from the scratch

    ```
    python train.py
    ```

+ restore from checkpoint

    ```
    python train.py --resume True --restore_path your_path
    ```

## Evaluate

+ evaluate from the default checkpoint
    
    ```
    python evaluate.py
    ```
    
+ evaluate from the specific checkpoint

    ```
    python evaluate.py --restore_path your_path
    ```
    
## Reference

+ [https://github.com/akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
+ [https://blog.csdn.net/Teeyohuang/article/details/79210525](https://blog.csdn.net/Teeyohuang/article/details/79210525)