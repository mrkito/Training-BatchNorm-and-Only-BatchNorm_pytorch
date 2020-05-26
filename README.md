# Training-BatchNorm-and-Only-BatchNorm_pytorch
Training-BatchNorm-and-Only-BatchNorm pytorch

dataset : https://github.com/fastai/imagenette

1) Train resnet18 
 ```
python .py
```
<img src="imgs/resnet_acc.png" width="700px" height="450px"/>
<img src="imgs/resnet_loss.png" width="700px" height="450px"/>

2) Train resnet18 without batchnorm 

<img src="imgs/resnet_no_bn_acc.png" width="700px" height="450px"/>
<img src="imgs/resnet_no_bn_loss.png" width="700px" height="450px"/>

3) Train resnet18 only batchnorm

<img src="imgs/resnet_bn_only_acc.png" width="700px" height="450px"/>
<img src="imgs/resnet_bn_only_loss.png" width="700px" height="450px"/>




# Citation

```
@article{
  title={Training BatchNorm and Only BatchNorm:On the Expressive Power of Random Features in CNNs},
  author={Jonathan Frankle, David J. Schwab , Ari S. Morcos},
  journal={arXiv preprint arXiv:2003.00152},
  year={2020}
}