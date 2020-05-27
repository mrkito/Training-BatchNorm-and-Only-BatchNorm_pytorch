# Training-BatchNorm-and-Only-BatchNorm_pytorch
Training-BatchNorm-and-Only-BatchNorm pytorch

dataset : https://github.com/fastai/imagenette
 ```
python train_loop.py
```
1) Train resnet18 

ResNet18 with an LR schedule and all the layers as trainable

<img src="imgs/accuracy01_resnet.png" width="700px" height="450px"/>

2) Train resnet18 without batchnorm 
ResNet18 with an LR schedule and all the layers as trainable without batchnorm 

<img src="imgs/accuracy01_resnet_no_bn.png" width="700px" height="450px"/>

3) Train resnet18 only batchnorm

ResNet18 with an LR schedule and only  batchnorm layers trainable

<img src="imgs/accuracy01_resnet_bn_only.png" width="700px" height="450px"/>


Result:

<img src="imgs/accuracy_val.png" width="700px" height="450px"/>
<img src="imgs/loss_val.png" width="700px" height="450px"/>


Visualization:

visualization.ipynb


# Citation

```
@article{
  title={Training BatchNorm and Only BatchNorm:On the Expressive Power of Random Features in CNNs},
  author={Jonathan Frankle, David J. Schwab , Ari S. Morcos},
  journal={arXiv preprint arXiv:2003.00152},
  year={2020}
}