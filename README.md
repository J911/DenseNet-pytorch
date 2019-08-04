# DenseNet-pytorch
DenseNet: Densely Connected Convolutional Networks 

A pytorch implementation of DenseNet([Huang, Gao, et al. “Densely connected convolutional networks.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1608.06993)) for CIFAR10

## Support & Requirements
- 🔥pytorch >= 0.4.0
- 🐍python 3.6.5 
- 📈tensorboardX 1.8

- ‼ multi GPU support!

## Training
git clone & change DIR
```bash
$ git clone https://github.com/J911/DenseNet-pytorch
$ cd DenseNet-pytorch
```
training
```bash
$ python train.py
```
### optional arguments:    
- --lr LR
- --resume RESUME
- --growth_rate GROWTH_RATE
- --theta THETA
- --dropout DROPOUT
- --batch_size BATCH_SIZE
- --batch_size_test BATCH_SIZE_TEST
- --momentum MOMENTUM
- --weight_decay WEIGHT_DECAY
- --epoch EPOCH
- --num_worker NUM_WORKER
- --logdir LOGDIR
